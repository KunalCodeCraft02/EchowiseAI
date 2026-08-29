"""
Groq extraction. Two independent halves:

  TEXT-ONLY SEMANTIC LAYER (LIVE) -- resolve_labels_semantically() / extract_narrative_findings()
  at the bottom of this file. These are what app/ocr/extractor.py calls. They receive TEXT ONLY,
  never an image: table structure is recovered geometrically in app/ocr/text_extraction.py, so
  the model's only job is language understanding.

  VISION EXTRACTION (DORMANT) -- extract_document() and everything it uses. Retired from the
  live pipeline in favour of local PaddleOCR geometry: sending a full page image cost ~6.7k
  tokens and routinely hit the free tier's rate limit, and geometry now recovers layout without
  it. Kept, tested, and unreferenced by app/ocr/pipeline.py.

WHY A VISION MODEL WAS USED INSTEAD OF PLAIN OCR (historical)
-------------------------------------------------------------
Real echo reports are dense multi-column tables. Plain OCR flattens them into a line soup where
"IVSd" and its value can end up far apart, so label->value association fails (measured:
Tesseract extracted 0 of 13 parameters from a real Chong Hua stress-echo report). A vision model
reads the *layout*, so it keeps each label attached to its own value. app/ocr/text_extraction.py
now solves the same problem locally by clustering PaddleOCR's bounding polygons into rows and
columns, which is deterministic and free.

WHAT THIS MODULE DOES NOT DO
-----------------------------
It deliberately does NOT decide which cardiac parameter a row is. It asks the model only to
TRANSCRIBE label/value pairs verbatim, and hands them to app/ocr/extractor.py as form_fields.
All the clinical logic -- synonym matching, indexed-variant routing (LAd/BSA vs LAd), unit
conversion to canonical units, physiological range rejection -- stays in that one tested place.
That keeps the model's job narrow (read the page) and generalizes to report layouts nobody has
seen yet.

CONFIDENCE
----------
Groq returns NO per-field confidence. This module therefore emits a neutral placeholder and
leaves real scoring to app/ocr/cross_validation.py, which derives confidence from whether an
independent OCR engine corroborates each value. Confidence is never fabricated here.
"""
import base64
import json
import re
import time
from typing import Dict, List, Optional

import cv2
import httpx
import numpy as np

from app.config import (
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_SEMANTIC_ENABLED,
    GROQ_SEMANTIC_MODEL,
    GROQ_SEMANTIC_TIMEOUT_SECONDS,
    GROQ_VISION_MODEL,
    GROQ_VISION_TIMEOUT_SECONDS,
)

# Neutral placeholder only. cross_validation.corroborate() overwrites this with a value derived
# from real corroboration evidence. Never treat this as a measured confidence.
_PLACEHOLDER_CONFIDENCE = 0.0

_PROMPT = """You are transcribing a 2D echocardiogram / stress-echo report image.

Extract EVERY measurement row you can see as a label/value pair.

Rules -- follow exactly:
1. Copy the label EXACTLY as printed, including any suffix. For example "LAd (AP)", "LAd/BSA",
   "EF (Simpson)", "IVSd", "SOV", "STJ". Never expand, translate, or normalize a label.
2. Copy the value EXACTLY as printed, including its unit if one is shown, e.g. "4.2 cm",
   "65 %", "1.62", "12 mL/m2".
3. Do NOT calculate, infer, convert, or guess any value. Transcribe only what is visibly
   printed. If a value is blank or unreadable, omit that row entirely.
4. Include qualitative findings too (e.g. label "Conclusion", value "Mild mitral regurgitation").
5. A label and its value that sit in different columns of the same row still belong together.

Return ONLY a JSON array, no prose and no markdown fence:
[{"label": "...", "value": "..."}, ...]"""


class GroqVisionError(Exception):
    """Any Groq vision failure (auth, quota, network, unparseable output). pipeline.py catches
    this and degrades to local OCR rather than failing the report."""


class GroqVisionUnavailable(GroqVisionError):
    """The configured model cannot accept images at all (retired id, or a text-only model).
    Distinct from a transient failure so the UI can tell the user to change GROQ_VISION_MODEL."""


def _encode_image(image: np.ndarray, max_dim: int = 1600) -> str:
    """JPEG-encode a page image to base64. Downscales very large pages to keep the request
    within Groq's payload limits while staying legible."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise GroqVisionError("Could not JPEG-encode page image for Groq.")
    return base64.b64encode(buf.tobytes()).decode()


def _strip_reasoning(text: str) -> str:
    """qwen3.6 emits <think>...</think> before its answer. Remove it (including an unclosed
    block, which happens when the model is cut off by max_tokens)."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def _extract_json_array(text: str) -> Optional[str]:
    """Pull the first balanced JSON array out of the response, tolerating markdown fences and
    trailing prose."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("[")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_groq_response(content: str) -> List[Dict[str, str]]:
    """Turn raw model output into [{label, value}] pairs. Raises GroqVisionError if the output
    cannot be recovered -- never returns partial garbage."""
    cleaned = _strip_reasoning(content)
    blob = _extract_json_array(cleaned)
    if blob is None:
        raise GroqVisionError("Groq returned no JSON array (model output was not usable).")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise GroqVisionError(f"Groq returned malformed JSON: {exc}") from exc
    if not isinstance(data, list):
        raise GroqVisionError("Groq JSON was not an array of label/value pairs.")

    pairs: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        value = item.get("value")
        value = "" if value is None else str(value).strip()
        if label and value:
            pairs.append({"label": label, "value": value})
    return pairs


def _retry_after_seconds(resp, detail: str) -> float:
    """How long Groq wants us to wait. Prefers the Retry-After header, else parses the
    'try again in 29.93s' hint out of the error message."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return min(float(header), 65.0)
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)s", detail or "")
    if m:
        try:
            return min(float(m.group(1)) + 1.0, 65.0)
        except ValueError:
            pass
    return 20.0


def _call_groq(image_b64: str, max_retries: int = 2) -> str:
    """POST one page image to Groq.

    Groq's free tier is token-rate-limited (8,000 TPM at time of writing) and a single
    full-page report image costs ~6,700 tokens -- so a 429 is an ordinary, expected event
    rather than an error. Honour the server's own retry hint instead of failing the report.
    """
    if not GROQ_API_KEY:
        raise GroqVisionError("GROQ_API_KEY is not set -- add it to backend/.env.")
    body = {
        "model": GROQ_VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        "temperature": 0,       # transcription must be deterministic, never creative
        # Groq counts prompt_tokens + max_tokens against the free tier's 8,000 TPM budget and
        # rejects the request with HTTP 413 if the sum exceeds it. A page image plus this
        # prompt costs ~2.7k, so keep the completion budget under ~5k to stay inside the
        # window while still leaving room for a long table transcription.
        "max_tokens": 5000,
        # Keep reasoning minimal -- this is verbatim transcription, not a reasoning task, and
        # thinking tokens are the main cause of truncation here.
        "reasoning_effort": "none",
    }

    last_detail = ""
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=body,
                timeout=GROQ_VISION_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            raise GroqVisionError(f"Groq request failed: {exc}") from exc

        if resp.status_code == 200:
            payload = resp.json()
            try:
                choice = payload["choices"][0]
            except (KeyError, IndexError, TypeError) as exc:
                raise GroqVisionError(f"Unexpected Groq response shape: {exc}") from exc
            # A truncated response has no closing </think>, so there is no JSON to parse --
            # fail loudly rather than silently extracting a partial table.
            if choice.get("finish_reason") == "length":
                raise GroqVisionError(
                    "Groq response was truncated (max_tokens reached) -- output not trustworthy."
                )
            return choice["message"]["content"]

        try:
            last_detail = resp.json().get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001 -- error body may not be JSON
            last_detail = resp.text[:200]

        if resp.status_code == 429 and attempt < max_retries:
            time.sleep(_retry_after_seconds(resp, last_detail))
            continue

        if resp.status_code in (400, 404) and (
            "must be a string" in last_detail or "model" in last_detail.lower()
        ):
            raise GroqVisionUnavailable(
                f"Groq model '{GROQ_VISION_MODEL}' cannot accept images "
                f"({last_detail[:120]}) -- set GROQ_VISION_MODEL to a multimodal model."
            )
        raise GroqVisionError(f"Groq HTTP {resp.status_code}: {last_detail}")

    raise GroqVisionError(f"Groq rate limit persisted after retries: {last_detail}")


# ===========================================================================
# TEXT-ONLY SEMANTIC LAYER
# ===========================================================================
# Everything below sends TEXT ONLY -- never an image. Structure detection was already solved
# geometrically in app/ocr/text_extraction.py, so the model's remaining job is purely language
# understanding: recognising that "LAd (AP)" means the same field as "LA Diameter", and reading
# a severity finding out of an English sentence.
#
# Both entry points are best-effort by contract: they raise GroqVisionError on any failure and
# app/ocr/extractor.py falls back to deterministic dictionary matching. Nothing here is ever
# required for the pipeline to produce a result.

_LABEL_PROMPT = """You map echocardiogram report labels onto a fixed list of canonical field keys.

CANONICAL KEYS (the ONLY values you may return):
{keys}

LABELS TO MAP:
{labels}

Rules -- follow exactly:
1. Return a canonical key only when the label means THE SAME measurement. "LAd (AP)" is the
   left atrial AP diameter, "EF (Simpson)" is the ejection fraction.
2. Return null when the label is not one of these measurements, or you are not sure. A null is
   always better than a wrong mapping -- this is clinical data.
3. A label carrying an indexed suffix ("/BSA", "index", "indexed", "(i)") is NEVER the base
   measurement. Map it to the matching *_Indexed key if one exists, otherwise null.
   "LVEDD/BSA" is NOT "LVIDd".
4. Do not invent keys. Do not return a key that is not in the list above.

Return ONLY a JSON array, no prose and no markdown fence:
[{{"label": "<label exactly as given>", "key": "<canonical key or null>"}}, ...]"""

_NARRATIVE_PROMPT = """You read qualitative findings out of the narrative sections of an
echocardiogram report (Conclusion / Impression / Findings / Valves / Summary / Interpretation).

CANONICAL KEYS (the ONLY values you may return):
{keys}

REPORT TEXT:
{blocks}

Rules -- follow exactly:
1. Extract descriptive findings written as prose, e.g. "Mild mitral regurgitation" -> key "MV",
   value "Mild Regurgitation". Keep the severity grader with the finding.
2. `value` must be a short descriptive phrase copied or lightly normalized from the text
   (e.g. "Normal", "Mild Regurgitation", "Moderately Dilated"). Never invent a number.
3. If the text states a measurement with a number, you may return it, with value_type "numeric".
   Otherwise value_type is "qualitative".
4. Report only what the text actually says. Omit anything you are unsure about.
5. Do not invent keys. Do not return a key that is not in the list above.

Return ONLY a JSON array, no prose and no markdown fence:
[{{"key": "...", "value": "...", "value_type": "numeric|qualitative", "evidence": "<the sentence>"}}, ...]"""


def _call_groq_text(prompt: str, max_tokens: int = 1500) -> str:
    """POST a text-only completion to Groq. No image, no retry loop -- a semantic lookup that
    fails is not worth stalling a report for; the caller degrades to dictionary matching."""
    if not GROQ_SEMANTIC_ENABLED:
        raise GroqVisionError("Groq semantic layer disabled (GROQ_SEMANTIC_ENABLED=0).")
    if not GROQ_API_KEY:
        raise GroqVisionError("GROQ_API_KEY is not set -- add it to backend/.env.")

    body = {
        "model": GROQ_SEMANTIC_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,   # clinical mapping must be deterministic, never creative
        "max_tokens": max_tokens,
    }
    try:
        resp = httpx.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json=body,
            timeout=GROQ_SEMANTIC_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise GroqVisionError(f"Groq semantic request failed: {exc}") from exc

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001 -- error body may not be JSON
            detail = resp.text[:200]
        raise GroqVisionError(f"Groq semantic HTTP {resp.status_code}: {detail}")

    try:
        choice = resp.json()["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroqVisionError(f"Unexpected Groq response shape: {exc}") from exc
    if choice.get("finish_reason") == "length":
        raise GroqVisionError("Groq semantic response was truncated -- output not trustworthy.")
    return choice["message"]["content"]


def _parse_json_array(content: str) -> List[Dict]:
    blob = _extract_json_array(_strip_reasoning(content))
    if blob is None:
        raise GroqVisionError("Groq returned no JSON array.")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise GroqVisionError(f"Groq returned malformed JSON: {exc}") from exc
    if not isinstance(data, list):
        raise GroqVisionError("Groq JSON was not an array.")
    return [item for item in data if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# PHI redaction -- applied to EVERYTHING leaving this machine
# ---------------------------------------------------------------------------
# The text-only calls below send OCR'd report text to a third-party API. That text routinely
# carries the patient's name, hospital number and date of birth alongside the measurements --
# none of which the model needs in order to map "LAd (AP)" to LA_Diameter or to read "mild
# mitral regurgitation" out of a sentence. Identity is stripped before the request so the
# clinical content still gets the benefit of the model while the identifiers stay local.
#
# This is redaction, not anonymisation: it removes the identifiers these report templates
# actually print. It is not a substitute for a BAA/DPA if real patient records are processed.
_ID_LABEL_RE = re.compile(
    r"(?im)^([ \t]*(?:patient\s*name|patient\s*id|name|hh\s*no\.?|id\s*no\.?|mrn|uhid|"
    r"date\s*of\s*birth|dob|birth\s*date|age\s*/\s*sex|referred\s*by|referring\s*physician|"
    r"ordering\s*physician|physician|sonographer|consultant|address|cell\s*no\.?|"
    r"tel(?:ephone)?|phone|lab\s*number|request\s*no|order\s*no|adm\s*no|ref\.?\s*by)"
    r"[ \t]*[:\-][ \t]*).+$"
)
# Long digit runs are record numbers, not measurements: no echo parameter has six digits.
_LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")


def redact_identifiers(text: str) -> str:
    """Strip patient-identifying text before it is sent to the Groq API."""
    if not text:
        return text
    text = _ID_LABEL_RE.sub(r"\1[REDACTED]", text)
    text = _LONG_NUMBER_RE.sub("[REDACTED]", text)
    text = _DATE_RE.sub("[DATE]", text)
    return text


def resolve_labels_semantically(labels: List[str], canonical_keys: List[str]) -> Dict[str, str]:
    """Map report label variants onto canonical dictionary keys.

    Called ONCE per document with every label the deterministic dictionary matcher could not
    resolve. Returns {original_label: canonical_key}; labels the model declined are simply
    absent. The caller re-validates every key against the dictionary and re-applies the
    disqualifying-suffix rules, so a bad answer here cannot reach storage.
    """
    labels = [l for l in dict.fromkeys(l.strip() for l in labels) if l]
    # A label that redaction alters carries an identifier (record number, date of birth). It is
    # never a cardiac parameter, so it is DROPPED rather than redacted -- sending "[REDACTED]"
    # would only get "[REDACTED]" back, and the caller looks results up by the original label.
    # Residual gap, stated plainly: a bare patient name sitting in a label cell has no syntactic
    # marker and is not detectable here. The narrative path, which is where report headers like
    # "Patient Name: ..." actually appear, is fully redacted.
    labels = [l for l in labels if redact_identifiers(l) == l]
    if not labels:
        return {}

    prompt = _LABEL_PROMPT.format(
        keys="\n".join(f"- {k}" for k in canonical_keys),
        labels="\n".join(f"- {l}" for l in labels),
    )
    mapping: Dict[str, str] = {}
    for item in _parse_json_array(_call_groq_text(prompt)):
        label = str(item.get("label") or "").strip()
        key = item.get("key")
        if label and key and str(key).strip().lower() not in ("null", "none", ""):
            mapping[label] = str(key).strip()
    return mapping


def extract_narrative_findings(blocks: List[str], canonical_keys: List[str]) -> List[Dict]:
    """Pull qualitative findings out of narrative prose.

    Returns [{"key", "value", "value_type", "evidence"}]. Severity/descriptor findings are as
    often written as a sentence as they are printed in a table, so this is where most
    qualitative values in a real report actually come from.
    """
    blocks = [b.strip() for b in (blocks or []) if b and b.strip()]
    if not blocks:
        return []

    prompt = _NARRATIVE_PROMPT.format(
        keys="\n".join(f"- {k}" for k in canonical_keys),
        # Redacted BEFORE truncation, so identifiers can never survive inside the sent prefix.
        blocks=redact_identifiers("\n\n".join(blocks))[:8000],  # 8000: free-tier TPM budget
    )
    findings: List[Dict] = []
    for item in _parse_json_array(_call_groq_text(prompt)):
        key = str(item.get("key") or "").strip()
        value = str(item.get("value") or "").strip()
        if not key or not value:
            continue
        vtype = str(item.get("value_type") or "qualitative").strip().lower()
        findings.append({
            "key": key,
            "value": value,
            "value_type": "numeric" if vtype == "numeric" else "qualitative",
            "evidence": str(item.get("evidence") or "").strip(),
        })
    return findings


def extract_document(images: List[np.ndarray]) -> Dict:
    """Run Groq vision over each page image and return the unified doc_result shape consumed by
    extractor.extract_parameters_structured()."""
    if not images:
        raise GroqVisionError("No page images to process.")

    form_fields: List[Dict] = []
    text_parts: List[str] = []
    errors: List[str] = []

    for page_no, image in enumerate(images, start=1):
        try:
            content = _call_groq(_encode_image(image))
            for pair in parse_groq_response(content):
                form_fields.append({
                    "key_text": pair["label"],
                    "key_confidence": _PLACEHOLDER_CONFIDENCE,
                    "value_text": pair["value"],
                    "value_confidence": _PLACEHOLDER_CONFIDENCE,
                    "page": page_no,
                })
                text_parts.append(f'{pair["label"]}: {pair["value"]}')
        except GroqVisionError as exc:
            errors.append(f"page {page_no}: {exc}")

    # Only a total failure is fatal; a bad page among several still yields usable data.
    if not form_fields:
        raise GroqVisionError("; ".join(errors) or "Groq returned no usable label/value pairs.")

    return {
        "full_text": "\n".join(text_parts),
        "tables": [],
        "form_fields": form_fields,
        "lines": [],
        "source_type": "groq_vision",
    }
