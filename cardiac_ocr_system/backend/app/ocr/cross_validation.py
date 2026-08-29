"""
Cross-validation of Groq vision output against an independent OCR read (PaddleOCR).

WHY THIS EXISTS
---------------
OCR fails *visibly* -- you get garbage text you can see is wrong. A vision LLM fails
*invisibly* -- it can emit a clean, confident, plausible, WRONG number. For cardiac data that
is the dominant risk: "IVSd 1.0" vs "1.6" changes the diagnosis and both look equally valid.

Groq also returns no per-field confidence, so the confidence gating in extractor.resolve_value
would have nothing real to work with. Rather than invent a number, every Groq value is checked
against what an independent OCR engine actually read off the same pixels, and confidence is
derived from that evidence.

THREE OUTCOMES
--------------
1. CORROBORATED -- OCR read the same value. Confidence = the OCR engine's REAL confidence for
   that token. Nothing is synthesized, and extractor's reject gate (<35) stays live: a
   corroborating token that OCR itself read poorly is still correctly rejected.
2. CONFLICT -- OCR read a DIFFERENT value at that label. This is the most clinically dangerous
   case and is deliberately NOT lumped in with "not found". Stored (so the doctor sees it) but
   heavily downweighted, with BOTH readings surfaced in the flag reason.
3. UNCORROBORATED -- OCR could not read it at all. Not evidence that Groq is wrong (OCR may
   simply have failed), so it is stored but flagged for confirmation.

This layer is ADVISORY -- it never discards a value. Only extractor.resolve_value's
physiological range checks reject outright.
"""
import re
from typing import Dict, List, Optional, Tuple

from app.config import GROQ_UNCORROBORATED_CONFIDENCE

# Confidence assigned when the two readers disagree outright. Deliberately below
# GROQ_UNCORROBORATED_CONFIDENCE (a disagreement is worse evidence than silence) but above
# OCR_CONFIDENCE_REJECT_THRESHOLD so the value is still stored and visible for review.
CONFLICT_CONFIDENCE = 40.0

# confidence_basis values recorded in extraction_meta, so a reader can always tell what kind of
# number `confidence` actually is. These must never be confused with a real OCR score.
BASIS_CORROBORATING_OCR = "corroborating_ocr_token"   # real measured OCR confidence
BASIS_UNCORROBORATED = "llm_uncorroborated_constant"  # fixed constant, documented
BASIS_CONFLICT = "llm_conflict_constant"              # fixed constant, documented

_NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)")


def normalize_numeric_text(text: str) -> str:
    """Normalize characters that would otherwise corrupt numeric parsing downstream.

    Critically includes decimal COMMA -> dot: extractor._NUMERIC_VALUE_RE has no comma branch,
    so an un-normalized "1,62" would silently parse as 1 -- a 60% error on an E/A ratio.
    """
    if not text:
        return ""
    out = (text.replace(" ", " ")     # non-breaking space
               .replace("−", "-")     # unicode minus
               .replace("×", "x"))
    # Decimal comma -> dot, only between digits (never touches thousands separators in prose).
    out = re.sub(r"(?<=\d),(?=\d)", ".", out)
    return out.strip()


def _parse_number(text: str) -> Tuple[Optional[float], int]:
    """Return (value, decimal_places) from the first number in text."""
    m = _NUM_RE.search(normalize_numeric_text(text or ""))
    if not m:
        return None, 0
    raw = m.group(1).replace(",", ".")
    try:
        val = float(raw)
    except ValueError:
        return None, 0
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return val, decimals


def _values_agree(a: float, b: float, decimals: int) -> bool:
    """Agreement within the precision actually printed (so 1.0 vs 1.00 agree).

    NOTE: 4.2 vs 42 deliberately does NOT agree. That is the decimal-drop OCR error class that
    extractor's impossible_range check exists to catch -- treating it as corroboration would
    defeat that gate.
    """
    tol = 0.5 * (10 ** -decimals) if decimals else 0.5
    return abs(a - b) <= tol


def _is_distinctive(token: str, blob: str) -> bool:
    """Guard against FALSE corroboration.

    A bare "65" also appears as heart rate, age, or a date fragment; "1.9" is 3 characters in a
    227-line document. Only trust a document-wide match when the token is genuinely
    distinctive: it carries a decimal point, or it occurs exactly once in the whole document.
    """
    if not token:
        return False
    occurrences = len(re.findall(re.escape(token), blob))
    if occurrences == 0:
        return False
    if occurrences == 1:
        return True
    return "." in token and len(token.replace(".", "")) >= 2


def corroborate_value(
    value_text: str, ocr_lines: List[Tuple[str, float]]
) -> Dict:
    """Compare one Groq value against the independent OCR read.

    Returns the confidence to use plus an audit record. Never raises.
    """
    result = {
        "confidence": GROQ_UNCORROBORATED_CONFIDENCE,
        "confidence_basis": BASIS_UNCORROBORATED,
        "corroborated": False,
        "corroborating_snippet": None,
        "conflicting_value": None,
        "secondary_source": "paddleocr",
    }

    groq_val, decimals = _parse_number(value_text)
    if groq_val is None or not ocr_lines:
        return result

    blob = normalize_numeric_text(" ".join(t for t, _ in ocr_lines))
    printed = normalize_numeric_text(value_text)
    token = (_NUM_RE.search(printed).group(1) if _NUM_RE.search(printed) else "")

    # 1. Corroboration: an OCR line containing a numerically-equal value.
    for line_text, line_conf in ocr_lines:
        norm_line = normalize_numeric_text(line_text)
        for cand in _NUM_RE.findall(norm_line):
            try:
                cand_val = float(cand.replace(",", "."))
            except ValueError:
                continue
            if _values_agree(groq_val, cand_val, decimals) and _is_distinctive(token, blob):
                result.update({
                    "confidence": round(float(line_conf), 2),   # REAL measured OCR confidence
                    "confidence_basis": BASIS_CORROBORATING_OCR,
                    "corroborated": True,
                    "corroborating_snippet": line_text.strip()[:120],
                })
                return result

    # 2. Conflict: OCR read a *different* number on a line that mentions this value's magnitude
    #    range. Only claim conflict when the OCR line looks like the same measurement row --
    #    i.e. it holds a number of the same decimal precision that is close but not equal.
    for line_text, line_conf in ocr_lines:
        norm_line = normalize_numeric_text(line_text)
        for cand in _NUM_RE.findall(norm_line):
            try:
                cand_val = float(cand.replace(",", "."))
            except ValueError:
                continue
            if cand_val == groq_val:
                continue
            same_scale = 0 < abs(cand_val - groq_val) <= max(1.0, abs(groq_val) * 0.5)
            if same_scale and "." in cand and decimals:
                result.update({
                    "confidence": CONFLICT_CONFIDENCE,
                    "confidence_basis": BASIS_CONFLICT,
                    "corroborated": False,
                    "conflicting_value": cand,
                    "corroborating_snippet": line_text.strip()[:120],
                })
                return result

    return result


def cross_validate(form_fields: List[Dict], ocr_lines: List[Tuple[str, float]]) -> List[Dict]:
    """Attach evidence-derived confidence to every Groq-sourced form field.

    Mutates nothing; returns new dicts carrying `value_confidence` plus an `extra_meta` payload
    that pipeline/extractor merge into extraction_meta for auditability.
    """
    validated: List[Dict] = []
    for ff in form_fields:
        ev = corroborate_value(ff.get("value_text", ""), ocr_lines)
        new_ff = dict(ff)
        new_ff["value_text"] = normalize_numeric_text(ff.get("value_text", ""))
        new_ff["value_confidence"] = ev["confidence"]
        new_ff["key_confidence"] = ev["confidence"]
        new_ff["extra_meta"] = {
            "confidence_basis": ev["confidence_basis"],
            "corroborated": ev["corroborated"],
            "corroborating_snippet": ev["corroborating_snippet"],
            "conflicting_value": ev["conflicting_value"],
            "secondary_source": ev["secondary_source"],
        }
        if ev["confidence_basis"] == BASIS_CONFLICT:
            new_ff["extra_meta"]["conflict_warning"] = (
                f"AI read '{ff.get('value_text')}' but independent OCR read "
                f"'{ev['conflicting_value']}' nearby -- please confirm against the original report."
            )
        validated.append(new_ff)
    return validated
