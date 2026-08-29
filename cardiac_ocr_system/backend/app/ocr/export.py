"""
Per-report JSON export.

Every processed report is written to its own file under config.EXTRACTION_JSON_DIR, whether it
succeeded or failed. This is the machine-readable record of what the pipeline actually read: the
value stored for each parameter, the confidence behind it, which engine and which snippet it came
from, and whether it was flagged for doctor review.

It is written from the SAME ExtractedField objects the database row is built from, so the file
can never drift from what the UI shows.

A failed extraction still produces a file -- one recording the failure and the raw OCR text --
because "we could not read this report" is exactly the case someone needs the audit trail for.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from app.config import EXTRACTION_JSON_DIR
from app.ocr.parameter_dict import PARAMETERS


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(text: Optional[str], fallback: str = "unknown") -> str:
    """Filename-safe fragment. Keeps the file identifiable without trusting user input."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in (text or "").strip())
    return (cleaned[:40].strip("_") or fallback)


def build_extraction_payload(report, results: Optional[Dict] = None,
                             doc_result: Optional[dict] = None) -> dict:
    """Assemble the JSON payload for one report.

    `results` is the {canonical: ExtractedField} map from extract_parameters_structured. When it
    is None (the pipeline failed before extraction ran) the parameter block is built from the
    stored row instead, so the file still describes what -- if anything -- landed in the DB.
    """
    meta = report.extraction_meta or {}
    confidences = report.confidence_scores or {}
    flagged = set(report.flagged_params or [])

    parameters = {}
    for canon, spec in PARAMETERS.items():
        db_field = spec["db_field"]
        field_meta = dict(meta.get(db_field) or {})
        ef = (results or {}).get(canon)

        entry = {
            "db_field": db_field,
            "value": getattr(ef, "value", None) if ef else getattr(report, db_field, None),
            "detected": None,          # filled below
            "confidence": round(float(confidences.get(db_field) or 0.0), 2),
            "flagged": db_field in flagged,
            "kind": spec["kind"],
            "canonical_unit": spec["canonical_unit"],
        }
        entry["detected"] = entry["value"] is not None
        # Carry the audit trail verbatim rather than reformatting it -- source/snippet/
        # flag_reason/conversion are the fields a reviewer needs to check a value against the
        # original page, and they already have settled meanings in extraction_meta.
        for key in ("value_type", "source", "source_snippet", "flag_reason",
                    "raw_detected_value", "raw_detected_unit", "conversion_applied"):
            if key in field_meta:
                entry[key] = field_meta[key]
        if ef is not None and ef.in_range is not None:
            entry["in_range"] = ef.in_range
        parameters[canon] = entry

    detected = [k for k, v in parameters.items() if v["detected"]]
    pipeline_meta = meta.get("_pipeline") or {}

    payload = {
        "report_uid": report.report_uid,
        "generated_at": _utc_now_iso(),
        "status": report.status,
        "progress_stage": report.progress_stage,
        "failure_reason": report.failure_reason,
        # Why the is-this-an-echo-report check ruled the way it did. Present only when the check
        # ran (i.e. nothing matched), and the single most useful line when someone disputes a
        # "wrong file" verdict.
        "document_check": pipeline_meta.get("document_check"),
        "patient": {
            "name": report.patient_name,
            "age": report.patient_age,
            "gender": report.patient_gender,
        },
        "source_file": report.original_filename,
        "engine": {
            "source_type": (doc_result or {}).get("source_type") or pipeline_meta.get("engine"),
            "quality_info": (doc_result or {}).get("quality_info") or pipeline_meta.get("quality"),
            "paddle_error": pipeline_meta.get("paddle_error"),
            "tesseract_error": pipeline_meta.get("tesseract_error"),
        },
        "summary": {
            "parameters_total": len(parameters),
            "parameters_detected": len(detected),
            "parameters_not_detected": len(parameters) - len(detected),
            "flagged_for_review": sorted(flagged),
        },
        "parameters": parameters,
        # The unfiltered read, so a reviewer can see exactly what the engine saw -- including the
        # unreadable regions that explain a blank field.
        "raw_ocr_text": report.raw_ocr_text or "",
    }
    return payload


def write_extraction_json(report, results: Optional[Dict] = None,
                          doc_result: Optional[dict] = None) -> Optional[Path]:
    """Write the payload to EXTRACTION_JSON_DIR/<uid>__<patient>.json. Returns the path.

    Never raises: an export failure must not lose a report that extracted successfully, so the
    error is swallowed and reported by returning None.
    """
    try:
        EXTRACTION_JSON_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{_safe_slug(report.report_uid, 'report')}__{_safe_slug(report.patient_name)}.json"
        path = EXTRACTION_JSON_DIR / name
        payload = build_extraction_payload(report, results, doc_result)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
    except Exception:  # noqa: BLE001 -- exporting is a side record, never the critical path
        return None
