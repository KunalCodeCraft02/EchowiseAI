"""
Google Document AI integration.

Replaces PaddleOCR/Tesseract for scanned PDFs and photographed reports with a processor
that understands table and form structure, not just flat text -- which is what lets the
matching engine in extractor.py read across a table row instead of guessing from a flat
line list.

Configuration (see app/config.py for where each value comes from in the GCP console):
  GCP_PROJECT_ID, GCP_LOCATION, DOCAI_PROCESSOR_ID, DOCAI_PROCESSOR_TYPE,
  DOCAI_TIMEOUT_SECONDS, and GOOGLE_APPLICATION_CREDENTIALS (read directly from the
  environment by google-auth -- no local constant needed for it).

Never crashes the caller: every failure mode (auth, network, quota, misconfigured processor)
is wrapped in DocAIError and raised, so app/ocr/pipeline.py can catch it and fall back to the
local OCR path rather than dying.
"""
from pathlib import Path
from typing import Dict, List, Optional

from app.config import (
    GCP_PROJECT_ID,
    GCP_LOCATION,
    DOCAI_PROCESSOR_ID,
    DOCAI_TIMEOUT_SECONDS,
)

# ---------------------------------------------------------------------------
# DORMANT MODULE
# ---------------------------------------------------------------------------
# Document AI is no longer the active engine -- app/ocr/groq_extractor.py replaced it after the
# GCP project stayed blocked on billing. This module is kept intact (not deleted) so the
# integration can be revived, but google-cloud-documentai is NO LONGER INSTALLED.
#
# All google.* imports are therefore done LAZILY inside the functions below, so that importing
# this module never fails. Calling into it without the package raises DocAIError, which
# pipeline.py already handles as a normal degradation.
#
# !! WARNING before re-enabling: `pip install google-cloud-documentai` upgrades protobuf to
# >=4.25.8, but paddlepaddle 2.6.2 pins protobuf<=3.20.2 on Windows. Installing Document AI
# WILL break PaddleOCR (and vice versa) -- they cannot coexist. Pick one.
# ---------------------------------------------------------------------------


def _import_documentai():
    """Lazily import the Document AI SDK, converting absence into a normal DocAIError."""
    try:
        from google.api_core.client_options import ClientOptions
        from google.api_core import exceptions as gax_exceptions
        from google.cloud import documentai
        return documentai, ClientOptions, gax_exceptions
    except ImportError as exc:
        raise DocAIError(
            "google-cloud-documentai is not installed -- Document AI is dormant; the active "
            "engine is Groq vision (app/ocr/groq_extractor.py). Reinstalling it would break "
            "PaddleOCR via a protobuf version conflict."
        ) from exc

_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

_CLIENT_INSTANCE: Optional["documentai.DocumentProcessorServiceClient"] = None


class DocAIError(Exception):
    """Raised for any Document AI failure (auth, network, quota, bad config). Callers should
    catch this specifically and fall back to the local OCR path -- never let it crash the
    pipeline."""


def mime_type_for_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    mime = _MIME_TYPES.get(suffix)
    if not mime:
        raise DocAIError(f"Unsupported file type for Document AI: '{suffix}'")
    return mime


def get_documentai_client():
    """Lazy singleton client, pinned to the configured region's regional endpoint (Document AI
    processors are region-specific, e.g. 'us-documentai.googleapis.com')."""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        documentai, ClientOptions, _ = _import_documentai()
        opts = ClientOptions(api_endpoint=f"{GCP_LOCATION}-documentai.googleapis.com")
        _CLIENT_INSTANCE = documentai.DocumentProcessorServiceClient(client_options=opts)
    return _CLIENT_INSTANCE


def _text_from_anchor(text_anchor, full_text: str) -> str:
    """Concatenate every text segment referenced by a text anchor. A layout can span multiple
    non-contiguous segments of the full document text."""
    if text_anchor is None or not text_anchor.text_segments:
        return ""
    parts = []
    for seg in text_anchor.text_segments:
        start = int(seg.start_index) if seg.start_index else 0
        end = int(seg.end_index) if seg.end_index else 0
        parts.append(full_text[start:end])
    return "".join(parts).strip()


def _confidence_from_layout(layout) -> float:
    """Document AI reports confidence on a 0.0-1.0 scale; this project uses 0-100 everywhere
    else (ExtractedField.confidence, report.confidence_scores). This is the single conversion
    point -- nothing downstream needs to know two scales exist."""
    if layout is None:
        return 0.0
    return float(layout.confidence) * 100.0


def _bounding_box_from_layout(layout) -> Optional[List[Dict[str, float]]]:
    if layout is None or layout.bounding_poly is None:
        return None
    verts = layout.bounding_poly.normalized_vertices
    if not verts:
        return None
    return [{"x": float(v.x), "y": float(v.y)} for v in verts]


def _cell_to_dict(cell, full_text: str) -> Dict:
    layout = cell.layout
    return {
        "text": _text_from_anchor(layout.text_anchor, full_text),
        "confidence": round(_confidence_from_layout(layout), 2),
        "bounding_box": _bounding_box_from_layout(layout),
    }


def _entity_page(entity) -> int:
    """Best-effort page number for an entity, from its page_anchor. Defaults to 1."""
    try:
        refs = entity.page_anchor.page_refs
        if refs:
            return int(refs[0].page) + 1
    except Exception:  # noqa: BLE001 -- page_anchor is optional metadata, never worth failing on
        pass
    return 1


def _entities_to_form_fields(document: "documentai.Document") -> List[Dict]:
    """Flatten document.entities (and any nested properties) into the unified form-field shape.

    A Custom Extractor's entity `type_` is the trained label (e.g. "lvidd") and `mention_text`
    is the value it read, which maps cleanly onto key_text/value_text. Confidence is the
    entity's own score, converted to this project's 0-100 scale -- still a real API-provided
    value, never synthesized.
    """
    out: List[Dict] = []

    def _walk(entity):
        key = (entity.type_ or "").strip()
        value = (entity.mention_text or "").strip()
        conf = round(float(entity.confidence) * 100.0, 2)
        if key and value:
            out.append({
                "key_text": key,
                "key_confidence": conf,
                "value_text": value,
                "value_confidence": conf,
                "page": _entity_page(entity),
            })
        for child in entity.properties or []:
            _walk(child)

    for entity in document.entities or []:
        _walk(entity)
    return out


def _document_to_result(document: "documentai.Document") -> Dict:
    """Convert a raw Document AI Document proto into the unified doc_result shape shared by
    every extraction source (Document AI / pdfplumber digital / local OCR fallback)."""
    full_text = document.text or ""

    tables: List[Dict] = []
    form_fields: List[Dict] = []
    lines: List = []

    for page_idx, page in enumerate(document.pages):
        page_number = int(page.page_number) if page.page_number else page_idx + 1

        for table in page.tables:
            rows = []
            for row in list(table.header_rows) + list(table.body_rows):
                cells = [_cell_to_dict(cell, full_text) for cell in row.cells]
                rows.append({"cells": cells})
            tables.append({"page": page_number, "rows": rows})

        for ff in page.form_fields:
            # field_name/field_value ARE Layout objects directly (not wrappers with a nested
            # .layout attribute) -- confirmed against the installed google-cloud-documentai
            # proto definitions.
            key_layout = ff.field_name if ff.field_name else None
            value_layout = ff.field_value if ff.field_value else None
            form_fields.append({
                "key_text": _text_from_anchor(key_layout.text_anchor if key_layout else None, full_text),
                "key_confidence": round(_confidence_from_layout(key_layout), 2),
                "value_text": _text_from_anchor(value_layout.text_anchor if value_layout else None, full_text),
                "value_confidence": round(_confidence_from_layout(value_layout), 2),
                "page": page_number,
            })

        for line in page.lines:
            text = _text_from_anchor(line.layout.text_anchor, full_text)
            if text:
                lines.append((text, round(_confidence_from_layout(line.layout), 2)))

    # A CUSTOM_EXTRACTOR processor does NOT populate page.form_fields -- it returns its trained
    # labels in document.entities instead. Map those into the same key/value shape so they feed
    # the highest-priority form-field matcher, making DOCAI_PROCESSOR_TYPE="CUSTOM_EXTRACTOR"
    # actually work rather than silently extracting nothing. FORM_PARSER leaves entities empty,
    # so this is a no-op there.
    form_fields.extend(_entities_to_form_fields(document))

    return {
        "full_text": full_text,
        "tables": tables,
        "form_fields": form_fields,
        "lines": lines,
        "source_type": "document_ai",
    }


def process_document(file_bytes: bytes, mime_type: str) -> Dict:
    """Send a document to the configured Document AI processor and return the unified
    doc_result shape. Raises DocAIError on any failure -- callers must catch this and fall
    back to the local OCR path rather than letting it propagate."""
    if not GCP_PROJECT_ID or not DOCAI_PROCESSOR_ID:
        raise DocAIError(
            "Document AI is not configured (GCP_PROJECT_ID / DOCAI_PROCESSOR_ID missing from "
            "environment) -- see app/config.py for where to find these in the GCP console."
        )

    documentai, _, gax_exceptions = _import_documentai()
    try:
        client = get_documentai_client()
        processor_name = client.processor_path(GCP_PROJECT_ID, GCP_LOCATION, DOCAI_PROCESSOR_ID)
        request = documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(content=file_bytes, mime_type=mime_type),
        )
        result = client.process_document(request=request, timeout=DOCAI_TIMEOUT_SECONDS)
        return _document_to_result(result.document)
    except DocAIError:
        raise
    except (gax_exceptions.GoogleAPICallError, gax_exceptions.RetryError) as exc:
        raise DocAIError(f"Document AI API call failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure must degrade, not crash
        raise DocAIError(f"Document AI processing failed: {exc}") from exc
