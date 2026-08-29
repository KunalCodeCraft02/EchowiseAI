"""
End-to-end pipeline: uploaded file -> structured cardiac parameters.
Updates a CardiacReport row's progress/status as it goes, so the frontend
progress bar can poll and reflect real stages.

ENGINE STRATEGY
---------------
  digital PDF  -> pdfplumber (free, exact, no OCR, no API call)
  scan / photo -> PaddleOCR with GEOMETRY (app/ocr/text_extraction.py) -- the only reader.
                  Bounding polygons are clustered into rows and columns locally, so table
                  structure is recovered from pixel coordinates rather than inferred by a
                  model. Groq is then used TEXT-ONLY (app/ocr/extractor.py) for label-variant
                  resolution and narrative findings; it never receives the page image.

No cloud Vision / Document AI is used anywhere -- the whole structure-detection path is local.

Every stage degrades rather than crashes: if PaddleOCR is unavailable we fall back to
Tesseract flat lines; if the text-only Groq call fails, extraction continues on deterministic
dictionary matching; if nothing can be read at all the report is marked failed with an
actionable message instead of a misleading empty "success".
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.ocr.preprocessing import (
    prepare_inputs,
    extract_structured_digital_pdf,
    load_image,
    preprocess_for_ocr,
    rasterize_pdf_pages,
    is_digital_pdf,
)
from app.ocr import text_extraction
from app.ocr.document_check import looks_like_cardiac_report
from app.ocr.export import write_extraction_json
from app.ocr.extractor import run_ocr, extract_parameters_structured, extract_impression_text
from app import models

STAGES = [
    (10, "Reading uploaded file"),
    (25, "Pre-processing image"),
    (45, "Reading report (OCR + table reconstruction)"),
    (65, "Matching parameters & reading narrative findings"),
    (85, "Validating values & scoring confidence"),
    (100, "Extraction complete"),
]

# Machine-readable cause of a failed extraction, stored on the report so the UI can react to the
# CAUSE rather than pattern-matching the human sentence next to it. The distinction that matters
# to the user is WRONG_FILE (they uploaded something that is not an echo report -- re-scanning it
# will never help) versus everything else (the right document, read badly).
FAILURE_WRONG_FILE = "wrong_file"        # readable text, but not a cardiac/echo report
FAILURE_NO_TEXT = "no_text"              # nothing could be read at all
FAILURE_NO_PARAMETERS = "no_parameters"  # looks like a report, but no value could be matched
FAILURE_ERROR = "error"                  # the pipeline itself raised

WRONG_FILE_MESSAGE = (
    "This file does not look like a 2D Echo / cardiac report -- no cardiac content was found "
    "in it. Please upload the correct report (a scanned or digital echo report), not a "
    "screenshot or an unrelated document."
)


def _update(db: Session, report: models.CardiacReport, progress: int, stage: str):
    report.progress = progress
    report.progress_stage = stage
    report.status = "processing" if progress < 100 else "extracted"
    db.add(report)
    db.commit()


def load_raw_pages(file_path: str) -> List[np.ndarray]:
    """Original, unprocessed page images.

    Both PaddleOCR and the vision model perform BETTER on the raw image than on the binarized
    one produced for Tesseract (measured on a real report: raw 227 lines @ 96.6% avg confidence
    vs binarized 221 @ 92.0%). Adaptive thresholding destroys the anti-aliasing these engines
    are trained on, so they must not be fed prepare_inputs()' Tesseract-oriented output.
    """
    if Path(file_path).suffix.lower() == ".pdf":
        return rasterize_pdf_pages(file_path)
    return [load_image(file_path)]


def _read_all_pages(images, reader) -> Tuple[Optional[dict], Optional[str]]:
    """Run one geometry reader over every page. Returns (merged_document, error)."""
    docs: List[dict] = []
    error: Optional[str] = None
    for img in images:
        try:
            docs.append(reader(img))
        except Exception as exc:  # noqa: BLE001 -- caller decides whether to try another reader
            error = str(exc)
            break
    if not docs:
        return None, error
    return text_extraction.merge_documents(docs), error


def _read_pages_with_geometry(images, info: dict) -> Optional[dict]:
    """Read every page WITH bounding-box geometry, so table reconstruction can run.

    PaddleOCR first (better on report scans), Tesseract second. Both preserve geometry, so
    losing PaddleOCR costs accuracy but NOT table structure -- which is the difference between a
    slightly worse read and a report that extracts nothing at all from a multi-column layout.
    Whatever went wrong is recorded in `info` so the failure message can name it.
    """
    document, paddle_error = _read_all_pages(images, text_extraction.run_paddle_with_geometry)
    if document is not None:
        info["engine"] = "paddleocr_geometry"
        return document
    if paddle_error:
        info["paddle_error"] = paddle_error

    def _tesseract_reader(img):
        # Unlike PaddleOCR (see load_raw_pages), Tesseract does markedly better on the
        # binarized/deskewed image than on the raw one -- measured on a real report scan, 12
        # usable lines vs 4.
        return text_extraction.run_tesseract_with_geometry(preprocess_for_ocr(img)[0])

    document, tess_error = _read_all_pages(images, _tesseract_reader)
    if document is not None:
        info["engine"] = "tesseract_geometry"
        return document
    if tess_error:
        info["tesseract_error"] = tess_error
    return None


def _flat_ocr_lines(images) -> Tuple[List[Tuple[str, float]], Optional[str]]:
    """Last-resort geometry-free read (Tesseract). No polygons means no table reconstruction,
    so only the flat-line fallback matcher can use this."""
    lines: List[Tuple[str, float]] = []
    error: Optional[str] = None
    for img in images:
        try:
            lines.extend(run_ocr(img))
        except Exception as exc:  # noqa: BLE001 -- this is itself a best-effort fallback
            error = str(exc)
            break
    return lines, error


def _short_error(text: Optional[str], limit: int = 160) -> str:
    """First line of an exception message, trimmed -- enough to act on, short enough for the UI."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    first = first.strip().rstrip(".")
    return first[:limit] + ("..." if len(first) > limit else "")


def _describe_ocr_failure(pipeline_meta: dict) -> str:
    """One actionable sentence for the report UI, instead of a raw stack trace."""
    info = pipeline_meta.get("_pipeline") or {}
    paddle_error = _short_error(info.get("paddle_error"))
    tess_error = _short_error(info.get("tesseract_error"))

    if paddle_error and tess_error:
        return (f"Both OCR engines failed (PaddleOCR: {paddle_error}; Tesseract: {tess_error}). "
                "No text could be extracted -- please enter the values manually.")
    if paddle_error or tess_error:
        return (f"The OCR engine failed ({paddle_error or tess_error}). No text could be "
                "extracted from this document -- please enter the values manually.")
    return ("No text could be read from this document. It may be blank, badly out of focus, or "
            "an unsupported scan -- please re-upload a clearer copy or enter the values manually.")


def _degraded_engine_hint(pipeline_meta: dict) -> str:
    """Explain a partial failure in terms the user can act on -- including WHY the better engine
    was not used, rather than only that it wasn't."""
    info = pipeline_meta.get("_pipeline") or {}
    engine = info.get("engine")
    paddle_error = _short_error(info.get("paddle_error"))

    if engine == "tesseract_geometry":
        because = f" (PaddleOCR could not start: {paddle_error})" if paddle_error else ""
        return ("Note: the report was read by Tesseract rather than PaddleOCR" + because +
                ". Table structure was still reconstructed, but text accuracy is lower -- "
                "please re-upload a higher-resolution scan, or enter the values manually.")
    if engine == "tesseract_flat_lines":
        because = f" ({paddle_error})" if paddle_error else ""
        return ("Note: the report was read without bounding-box geometry" + because +
                ", so table columns could not be reconstructed -- which is why a dense "
                "multi-column layout extracts poorly.")
    return "Please re-upload a higher-resolution scan, or enter the values manually."


def _readable_texts(doc_result: dict) -> List[str]:
    """Every fragment of text the engines produced, for the is-this-a-report check.

    The table grid is flattened in as well: on a scan whose narrative half is unreadable, the
    only surviving cardiac vocabulary can be sitting in table cells.
    """
    texts = [
        doc_result.get("raw_text") or "",
        doc_result.get("full_text") or "",
        "\n".join(doc_result.get("narrative_blocks") or []),
    ]
    for row in doc_result.get("table_grid") or []:
        texts.append(" ".join(str(cell.get("text") or "") for cell in row if isinstance(cell, dict)))
    return texts


def _fail(db: Session, report: models.CardiacReport, reason: str, message: str,
          results=None, doc_result=None):
    """Mark the report failed with a cause AND a sentence, then export the audit record.

    Failures are exported like successes: "we could not read this" is precisely the case someone
    needs the audit trail for.
    """
    report.status = "failed"
    report.progress = 100
    report.failure_reason = reason
    report.progress_stage = f"Failed: {message}"
    db.add(report)
    db.commit()
    write_extraction_json(report, results, doc_result)


def _build_doc_result(file_path: str, pipeline_meta: dict) -> dict:
    """Run the local engine chain for a scanned/photographed report -> unified doc_result."""
    images = load_raw_pages(file_path)
    info: dict = {}

    document = _read_pages_with_geometry(images, info)

    if document is not None:
        info["quality"] = document.get("quality_info")
        narrative_blocks = document.get("narrative_blocks") or []
        pipeline_meta["_pipeline"] = info
        return {
            # Narrative sections drive prose findings; the grid drives the table path. raw_text
            # stays UNFILTERED and is what the preview page shows.
            "full_text": "\n\n".join(narrative_blocks),
            "raw_text": document.get("raw_text", ""),
            "ocr_only_text": document.get("raw_text", ""),
            "table_grid": document.get("table_grid") or [],
            "narrative_blocks": narrative_blocks,
            "quality_info": document.get("quality_info") or {},
            "tables": [],
            "form_fields": [],
            "lines": document.get("lines") or [],
            "source_type": info.get("engine", "geometry"),
        }

    # Both geometry readers failed. Last resort: flat text with no boxes at all.
    ocr_lines, fallback_error = _flat_ocr_lines(images)
    if fallback_error:
        info.setdefault("tesseract_error", fallback_error)
    if ocr_lines:
        info["engine"] = "tesseract_flat_lines"
    if info:
        pipeline_meta["_pipeline"] = info

    ocr_text = "\n".join(t for t, _ in ocr_lines)
    return {
        "full_text": ocr_text,
        "raw_text": ocr_text,
        "ocr_only_text": ocr_text,
        "table_grid": [],
        "narrative_blocks": [],
        "tables": [],
        "form_fields": [],
        "lines": ocr_lines,
        "source_type": info.get("engine", "none"),
    }


def process_report(db: Session, report: models.CardiacReport, file_path: str):
    """
    Runs the full pipeline synchronously and writes results directly onto the
    CardiacReport row (one row per report, holding every extracted parameter).
    """
    try:
        _update(db, report, *STAGES[0])
        pipeline_meta: dict = {}

        suffix = Path(file_path).suffix.lower()
        digital = suffix == ".pdf" and is_digital_pdf(file_path)
        _update(db, report, *STAGES[1])

        if digital:
            # Fast path: a digitally-generated PDF already has a real text layer -- exact,
            # free, and no API call needed.
            doc_result = extract_structured_digital_pdf(file_path)
            _update(db, report, *STAGES[2])
            results, raw_text = extract_parameters_structured(doc_result)
            # Some "digital" PDFs carry a junk embedded text layer; if nothing matched, fall
            # through to the vision/OCR chain rather than reporting an empty report.
            if not any(ef.value is not None for ef in results.values()):
                doc_result = _build_doc_result(file_path, pipeline_meta)
        else:
            _update(db, report, *STAGES[2])
            doc_result = _build_doc_result(file_path, pipeline_meta)

        _update(db, report, *STAGES[3])
        results, raw_text = extract_parameters_structured(doc_result)
        _update(db, report, *STAGES[4])

        confidence_scores = {}
        flagged = []
        extraction_meta = dict(pipeline_meta)
        for canon, ef in results.items():
            setattr(report, ef.db_field, ef.value)
            confidence_scores[ef.db_field] = ef.confidence
            extraction_meta[ef.db_field] = ef.meta
            if ef.flagged:
                flagged.append(ef.db_field)

        # Resolve BSA and compute downstream indexed fields
        from app.ocr.bsa import resolve_bsa, compute_indexed_fields
        params_dict = {f: getattr(report, f) for f in models.CardiacReport.PARAM_FIELDS}
        bsa_val, bsa_source_key, bsa_tag = resolve_bsa(params_dict)
        report.bsa = str(bsa_val) if bsa_val is not None else (report.bsa or "")
        report.bsa_source = bsa_source_key

        indexed = compute_indexed_fields(params_dict, bsa_val)
        for idx_field in ("lvidd_indexed_value", "la_diameter_indexed_value", "lv_mass"):
            if bsa_val is not None and indexed.get(idx_field):
                if not getattr(report, idx_field):
                    setattr(report, idx_field, indexed[idx_field])
            elif bsa_val is None:
                # BSA missing: clear indexed value and flag for manual BSA entry
                setattr(report, idx_field, "")
                if idx_field not in flagged:
                    flagged.append(idx_field)

        # The preview page must show what OCR actually read, unfiltered -- see the raw_ocr_text
        # note in models.py.
        report.raw_ocr_text = doc_result.get("raw_text") or doc_result.get("ocr_only_text") or raw_text
        # The report's own concluding prose, stored verbatim for the Impression card.
        # raw_ocr_text only: it is the unfiltered, line-for-line read, so what the card shows is
        # what the page said. None when the report has no Impression/Conclusion heading, which
        # hides the card rather than inventing a conclusion.
        report.impression_text = extract_impression_text(report.raw_ocr_text)
        report.confidence_scores = confidence_scores
        report.flagged_params = flagged
        report.extraction_meta = extraction_meta
        report.doctor_confirmed = []

        # SAFETY: never present a total extraction failure as a successful-but-empty report.
        # Every field rendering "Not detected" with 0 flagged reads, in a clinical UI, as "this
        # report genuinely contains none of these values" rather than "extraction did not run".
        got_any_text = bool(
            (doc_result.get("full_text") or "").strip()
            or (doc_result.get("raw_text") or "").strip()
            or doc_result.get("lines") or doc_result.get("tables")
            or doc_result.get("table_grid") or doc_result.get("form_fields")
        )
        if not got_any_text:
            _fail(db, report, FAILURE_NO_TEXT, _describe_ocr_failure(pipeline_meta),
                  results, doc_result)
            return

        if not any(ef.value is not None for ef in results.values()):
            # Nothing matched. Before blaming the scan, establish whether this is an echo report
            # at all -- telling someone to re-scan their WhatsApp screenshot at a higher
            # resolution sends them round a loop that cannot succeed.
            verdict = looks_like_cardiac_report(*_readable_texts(doc_result))
            extraction_meta.setdefault("_pipeline", {})["document_check"] = {
                "is_report": verdict.is_report,
                "score": verdict.score,
                "matched_terms": verdict.matched[:20],
            }
            report.extraction_meta = dict(extraction_meta)

            if not verdict.is_report:
                _fail(db, report, FAILURE_WRONG_FILE, WRONG_FILE_MESSAGE, results, doc_result)
                return

            _fail(db, report, FAILURE_NO_PARAMETERS,
                  "text was read but no cardiac parameters could be recognised. "
                  + _degraded_engine_hint(pipeline_meta),
                  results, doc_result)
            return

        _update(db, report, *STAGES[5])
        write_extraction_json(report, results, doc_result)

    except Exception as exc:
        report.status = "failed"
        report.failure_reason = FAILURE_ERROR
        report.progress_stage = f"Failed: {exc}"
        db.add(report)
        db.commit()
        raise
