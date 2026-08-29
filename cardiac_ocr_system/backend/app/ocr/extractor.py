"""
STAGE 2 -- SEMANTIC EXTRACTION
==============================
Turns the geometrically-reconstructed document from app/ocr/text_extraction.py (Stage 1) into
canonical cardiac parameters. Stage 1 already solved STRUCTURE from pixel geometry, so nothing
here has to guess layout -- this stage does normalization and language understanding only.

MATCHING PRIORITY (most-reliable source wins)
---------------------------------------------
  (a) form_fields  -- exact/near-exact normalized key match against synonyms. Used by the
                      digital-PDF path and any engine that emits label/value pairs directly.
  (b) table_grid   -- Stage 1's reconstructed grid. For each row independently, the label cell
                      is identified and ONLY that row's remaining cells are considered as its
                      value. Never another row, never leftward -- which is what structurally
                      prevents row-shift and label/value column-swap bugs rather than hoping a
                      model notices them.
  (c) tables       -- pdfplumber's own cell extraction (digital PDFs), same row-local rule.
  (d) narrative    -- qualitative findings read out of Conclusion/Impression/Findings prose.
                      Fills a field ONLY when (a)-(c) left it empty or below
                      OCR_CONFIDENCE_FLAG_THRESHOLD. Table always wins when both exist.
  (e) flat lines   -- last resort fuzzy substring scan, tagged "flat_line_fallback".

NUMERIC *AND* QUALITATIVE
-------------------------
A cell containing a word instead of a number is a valid extraction, not a miss: 2D-Echo reports
routinely print "Normal" / "Mild" / "Enlarged" / "Mild Regurgitation" where a number could have
gone. Every extraction is tagged value_type "numeric" or "qualitative", and the two take
deliberately different routes through resolve_value():
  numeric      -> unit detection -> conversion to canonical_unit -> impossible_range hard
                  rejection -> valid_range soft flag
  qualitative  -> stored verbatim. NO unit handling, NO range checking, no unit/conversion
                  metadata at all -- none of it is meaningful for a word, and running a word
                  through numeric validation could only ever discard a legitimate finding.

GROQ (TEXT ONLY, NEVER THE IMAGE)
---------------------------------
The deterministic dictionary matcher runs first -- it is free, offline and tested. Only labels
it cannot resolve are batched into a single text-only Groq call per document
(groq_extractor.resolve_labels_semantically), which maps real-world variants such as "LAd (AP)"
onto canonical keys. Everything Groq returns is re-validated against the dictionary and
re-checked against the disqualifying-suffix rules, so only canonical keys ever leave this
stage. If Groq is disabled or fails, extraction degrades to dictionary matching alone.
"""
import re
import difflib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Sequence, Any

import numpy as np

from app.ocr.parameter_dict import (
    DESCRIPTOR_TERMS,
    LESION_TERMS,
    PARAMETERS,
    QUALITATIVE_TERMS,
    SEVERITY_TERMS,
    normalized_key,
    sorted_synonym_index,
)
from app.config import LABEL_MATCH_THRESHOLD, TESSERACT_CMD, OCR_CONFIDENCE_FLAG_THRESHOLD, OCR_CONFIDENCE_REJECT_THRESHOLD
from app.predictor import normalize_v4 as N

_SYNONYM_INDEX = sorted_synonym_index()

# Unit tokens recognised after a number. ORDER MATTERS -- this is a regex alternation, so the
# longest form must come first or a prefix wins: without "cm2" ahead of "cm", the area
# "9.23 cm2" parsed as 9.23 CENTIMETRES and an atrial area was stored as a linear dimension.
# Likewise "cm/sec" must precede "cm/s", which must precede "cm".
_UNIT_ALTERNATION = (
    r"cm/sec|cm/s|m/sec|m/s|mmhg|mm\s?hg|cm2|cm²|cm/m2|ml/m2|m2|m²|kg|lbs|ml|cm|mm|%"
)

# Spelled-out and superscript unit forms normalized onto the keys used above.
_UNIT_ALIASES = {
    "cm/sec": "cm/s", "m/sec": "m/s", "mm hg": "mmhg", "cm²": "cm2", "m²": "m2",
}


_NUMERIC_VALUE_RE = re.compile(
    r"(\d{1,3}\.?\d{0,2})\s*(" + _UNIT_ALTERNATION + r")?", re.IGNORECASE
)


def normalize_unit(raw: Optional[str]) -> Optional[str]:
    """Fold a detected unit token onto the canonical spelling used by the parameter dictionary."""
    if not raw:
        return None
    key = " ".join(raw.strip().lower().split())
    return _UNIT_ALIASES.get(key, key)

# --- Printed reference ranges are not patient values -------------------------------------
# Almost every report template prints the normal range immediately beside the measurement --
# "Diastole (LVIDd)  Not detected  3.7-5.6 cm", "LVEF (est)  55%  >50%", "IVS (D) 1.3 0.6-1.1 cm".
# A naive "first number in the string" read stores the range's LOWER BOUND as the measurement,
# which is far worse than reading nothing: it is a plausible, in-range, entirely fictional value.
# Measured against tests/fixtures/ground_truth.json this was the single largest source of false
# positives (Ao_Root 2.0, LA_Diameter 1.9, IVSd 0.6, PWd 0.6 -- every one a range bound).
#
# Only DASH ranges and COMPARATORS are treated as ranges. A slash is deliberately NOT a range
# separator: "TJV 2.3/22 m/sec" and "PDA PG 73/10 mmHg" are real paired measurements, and BP
# "119/81" is a different field entirely.
_COMPARATOR_CHARS = "<>≤≥"          # < > <= >=
_RANGE_DASHES = "-‐‑‒–—"  # ASCII hyphen + unicode hyphens/dashes


def _scan_back(text: str, i: int) -> int:
    while i >= 0 and text[i] == " ":
        i -= 1
    return i


def _scan_fwd(text: str, i: int) -> int:
    while i < len(text) and text[i] == " ":
        i += 1
    return i


def _num_ending_at(text: str, i: int) -> Optional[float]:
    """Parse the number whose last character is text[i], scanning leftwards."""
    j = i
    while j >= 0 and (text[j].isdigit() or text[j] == "."):
        j -= 1
    try:
        return float(text[j + 1:i + 1])
    except ValueError:
        return None


def _num_starting_at(text: str, i: int) -> Optional[float]:
    """Parse the number whose first character is text[i], scanning rightwards."""
    j = i
    while j < len(text) and (text[j].isdigit() or text[j] == "."):
        j += 1
    try:
        return float(text[i:j])
    except ValueError:
        return None


def _reference_marker_precedes(text: str, start: int) -> bool:
    """True when an "F:" / "M:" normal-range marker introduces the numeric run at `start`."""
    i = start - 1
    while i >= 0 and (text[i].isdigit() or text[i] in ". ±+"):
        i -= 1
    if i < 0 or text[i] != ":":
        return False
    marker = _scan_back(text, i - 1)
    return marker >= 0 and text[marker] in "fmFM"


def _is_reference_range_context(text: str, start: int, end: int) -> bool:
    """True when the number at text[start:end] is part of a printed range, not a measurement.

    Three shapes, matching what the real templates actually print:
      ">50%" / "< 40"      a comparator immediately to the left
      "3.7-5.6"            a dash to the left, over a SMALLER number  -> this is the UPPER bound
      "3.7-5.6"            a dash to the right, over a LARGER number  -> this is the LOWER bound

    Both dash tests require the pair to be strictly ASCENDING, because a printed range always
    is. That is what separates a range from a negative number sitting in the next column: the
    paediatric template prints "LVIDS -15 -0.41" (value 15 mm, Z-score -0.41), and without the
    ascending test the "15 -0.41" span reads as the range "15-0.41" and the real measurement is
    thrown away. It also keeps "LVIDD -26" (label, separator, value) readable as 26.
    """
    if _reference_marker_precedes(text, start):
        return True

    left = _scan_back(text, start - 1)
    if left >= 0:
        if text[left] in _COMPARATOR_CHARS:
            return True
        if text[left] in _RANGE_DASHES:
            before_dash = _scan_back(text, left - 1)
            if before_dash >= 0 and text[before_dash].isdigit():
                lower = _num_ending_at(text, before_dash)
                this = _num_starting_at(text, start)
                if lower is not None and this is not None and lower < this:
                    return True
    # "F:" / "M:" head the Normal(F/M) reference columns on Siemens-style templates, so a number
    # introduced by one is a sex-specific normal range, never the patient's measurement. Without
    # this, a row whose value cell is blank ("II.Right Ventricle", continued on page 2) read its
    # own normal range and reported RV_Size for a measurement the report never took.
    #
    # The scan skips back over the WHOLE numeric run, not just one number: OCR mangles these
    # cells ("F: 2.3+-0.3 cm" came back as "F: 2.33.1 cm"), and blocking only the first number
    # left the "1 cm" tail behind to be stored instead.
    right = _scan_fwd(text, end)
    if right < len(text) and text[right] in _RANGE_DASHES:
        after_dash = _scan_fwd(text, right + 1)
        if after_dash < len(text) and text[after_dash].isdigit():
            upper = _num_starting_at(text, after_dash)
            this = _num_starting_at(text, start)
            if upper is not None and this is not None and upper > this:
                return True
    return False


def find_measurement(text: str) -> Optional[re.Match]:
    """First numeric match in `text` that is a measurement rather than a reference range.

    Returns None when the text contains only range/comparator numbers -- which is the correct
    outcome for a blank value cell whose neighbour holds the normal range.
    """
    for match in _NUMERIC_VALUE_RE.finditer(text or ""):
        if not match.group(1):
            continue
        if _is_reference_range_context(text, match.start(1), match.end()):
            continue
        if _is_inside_identifier(text, match):
            continue
        return match
    return None


def _is_inside_identifier(text: str, match: re.Match) -> bool:
    """True when the digits are part of a name, not a measurement.

    Report labels embed digits constantly -- "LA A4Cs", "Ejection fraction (2D)", "LAVI4 (MOD)",
    "SC2000RM2". Reading the first number in the text after a label match then yields "4 cm" for
    LA_Diameter (a real failure from tests/fixtures) instead of the measurement further along.

    The test is that the digits are followed immediately by a LETTER while no unit was matched.
    When a unit IS matched the token is a measurement whatever follows it, which keeps "2.1cms"
    and "0.8cm" readable -- both real OCR reads where the unit is glued to the number.
    """
    if match.group(2):
        return False
    after = match.end(1)
    return after < len(text) and text[after].isalpha()

VALUE_TYPE_NUMERIC = "numeric"
VALUE_TYPE_QUALITATIVE = "qualitative"

# Narrative-derived findings are inherently lower-trust than a structured table/form-field
# read (they're inferred from prose) -- fixed above REJECT so they're never silently dropped,
# but always below FLAG so they always surface for doctor confirmation.
_NARRATIVE_CONFIDENCE = 50.0

_UNIT_CONVERSION_FACTORS = {
    ("mm", "cm"): 0.1, ("cm", "mm"): 10.0,
    ("cm/s", "m/s"): 0.01, ("m/s", "cm/s"): 100.0,
    ("m", "cm"): 100.0, ("lbs", "kg"): 0.45359237,
}
_UNIT_DISPLAY = {"cm": "cm", "mm": "mm", "%": "%", "mmhg": "mmHg", "": "", None: "", "cm/m2": "cm/m2",
                 "m/s": "m/s", "cm/s": "cm/s", "cm2": "cm2", "m2": "m²", "kg": "kg", "lbs": "lbs"}


# Narrative sections are located by these headings. Valve/severity vocabulary now comes from
# parameter_dict (SEVERITY_TERMS / LESION_TERMS / DESCRIPTOR_TERMS) so the table path and the
# narrative path recognise exactly the same set of qualitative terms.
_HEADING_KEYWORDS = ("conclusion", "impression", "findings", "valve", "valves", "summary",
                     "interpretation")


@dataclass
class ExtractedField:
    canonical: str
    db_field: str
    value: Optional[str] = None
    confidence: float = 0.0
    source_line: str = ""
    in_range: Optional[bool] = None
    flagged: bool = False
    source: str = "flat_line_fallback"
    # "numeric" | "qualitative" | None (nothing was stored). Decides which validation path the
    # value took, and is surfaced in extraction_meta so the UI/doctor can tell a measured
    # number from a descriptive finding.
    value_type: Optional[str] = None
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Value-type classification
# ---------------------------------------------------------------------------
def _has_number(text: str) -> bool:
    match = _NUMERIC_VALUE_RE.search(text or "")
    return bool(match and match.group(1))


def looks_qualitative(text: str) -> bool:
    """True when a cell/phrase reads as a clinical descriptor rather than a measurement.

    Deliberately vocabulary-driven rather than "anything without digits": OCR garbage like
    "l~," must NOT be promoted to a qualitative finding just because it failed to parse as a
    number. A value only becomes qualitative if it actually says something clinical.
    """
    norm = normalized_key(text)
    if not norm:
        return False
    return any(re.search(r"\b" + re.escape(term) + r"\b", norm) for term in QUALITATIVE_TERMS)


_SEGMENT_PATTERNS = {
    "basal": re.compile(r"\b(?:basal|base)\b\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm)?", re.IGNORECASE),
    "mid": re.compile(r"\b(?:mid|middle)\b\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm)?", re.IGNORECASE),
    "apical": re.compile(r"\b(?:apical|apex)\b\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*(mm|cm)?", re.IGNORECASE),
}


def extract_segmental_measurements(text: str) -> Optional[Dict[str, Any]]:
    """Detect when IVS/PW measurements are reported in segmental format (Basal, Mid, Apical)
    e.g. "IVS – Basal: 14mm; Mid: 20mm; Apical: 22mm" or "PW – Basal: 12mm; Mid: 18mm; Apical: 20mm".
    Returns dict with:
      - 'segments': {'basal': '14 mm', 'mid': '20 mm', 'apical': '22 mm'}
      - 'segment_values_mm': {'basal': 14.0, 'mid': 20.0, 'apical': 22.0, 'max': 22.0}
      - 'basal_formatted': '14 mm'
    or None if fewer than 2 segments are found.
    """
    if not text:
        return None
    raw_found: Dict[str, str] = {}
    mm_found: Dict[str, float] = {}

    for seg_key, pattern in _SEGMENT_PATTERNS.items():
        m = pattern.search(text)
        if m:
            num_str, unit = m.group(1), m.group(2)
            try:
                num = float(num_str)
                unit_norm = (unit or "").lower().strip()
                if unit_norm == "cm" or (not unit_norm and num < 3.5):
                    num_mm = round(num * 10.0, 1)
                else:
                    num_mm = round(num, 1)
                raw_found[seg_key] = f"{num_mm:g} mm"
                mm_found[seg_key] = num_mm
            except ValueError:
                continue

    if len(raw_found) >= 2 and "basal" in raw_found:
        max_val = max(mm_found.values())
        mm_found["max"] = max_val
        return {
            "segments": raw_found,
            "segment_values_mm": mm_found,
            "basal_formatted": raw_found["basal"],
        }
    return None


def classify_value_type(text: str, kind: str) -> Optional[str]:
    """Decide how a raw value string should be handled. None => not a usable value at all."""
    if not (text or "").strip():
        return None
    if kind == "text":
        return VALUE_TYPE_QUALITATIVE if looks_qualitative(text) else None
    if _has_number(text):
        return VALUE_TYPE_NUMERIC
    # If kind is numeric, pure negation words are never qualitative findings or numbers
    if kind == "numeric":
        low = text.strip().lower()
        if re.search(r"\b(no|none|nil|not|absent|without)\b", low):
            return None
    if looks_qualitative(text):
        return VALUE_TYPE_QUALITATIVE
    return None


# ---------------------------------------------------------------------------
# OCR engines (local fallback, used when Document AI itself is unavailable)
# ---------------------------------------------------------------------------
def _run_paddle_ocr(image: np.ndarray) -> List[Tuple[str, float]]:
    from paddleocr import PaddleOCR  # imported lazily; optional dependency

    global _PADDLE_INSTANCE
    try:
        _PADDLE_INSTANCE
    except NameError:
        _PADDLE_INSTANCE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    result = _PADDLE_INSTANCE.ocr(image, cls=True)
    lines = []
    for page in result or []:
        # PaddleOCR 2.7.x returns [None] for a page where nothing was detected (e.g. a blank
        # scan). Iterating that raises TypeError, which run_ocr's except would swallow into a
        # silent, misleading downgrade to Tesseract -- so guard it explicitly.
        if not page:
            continue
        for box, (text, conf) in page:
            lines.append((text, float(conf) * 100.0))
    return lines


def _run_tesseract_ocr(image: np.ndarray) -> List[Tuple[str, float]]:
    import pytesseract

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    n = len(data["text"])

    # Group words back into lines using (block_num, par_num, line_num), keeping
    # per-line confidence as the average of its constituent words. This
    # preserves the row-based label:value layout that the report is printed in.
    lines_map: Dict[Tuple[int, int, int], List[Tuple[str, float]]] = {}
    for i in range(n):
        word = data["text"][i].strip()
        if not word:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines_map.setdefault(key, []).append((word, conf))

    lines = []
    for key in sorted(lines_map.keys()):
        words = lines_map[key]
        text = " ".join(w for w, _ in words)
        avg_conf = sum(c for _, c in words) / len(words)
        lines.append((text, avg_conf))
    return lines


def run_ocr(image: np.ndarray) -> List[Tuple[str, float]]:
    """Try PaddleOCR first; gracefully fall back to Tesseract if unavailable.

    In practice PaddleOCR is no longer installable alongside Document AI (paddlepaddle pins
    protobuf<=3.20.2 on Windows, google-cloud-documentai requires protobuf>=4), so this
    normally falls straight through to Tesseract. The Paddle branch is kept so an environment
    that does have a working PaddleOCR (e.g. Linux, or a separate worker) still uses it.
    """
    try:
        return _run_paddle_ocr(image)
    except Exception:
        return _run_tesseract_ocr(image)


# ---------------------------------------------------------------------------
# Flat-line label matching (used by the last-resort fallback path)
# ---------------------------------------------------------------------------
def _fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0


def _find_label_match(line: str) -> Optional[Tuple[str, str, int, int]]:
    """Return only the single best (first/longest) label match on a line."""
    matches = _find_all_label_matches(line)
    return matches[0] if matches else None


# A measurement glued straight onto its label by OCR: a decimal number, or an integer carrying
# a unit. "0.8cm", "3.4 cm", "73%", "22 mmHg" all qualify; a bare "4" does not.
_GLUED_MEASUREMENT_RE = re.compile(r"\d+\.\d|\d+\s*(?:cm|mm|mmhg|%)", re.IGNORECASE)

_HYPHENS = str.maketrans({c: " " for c in "-‐‑‒–—"})


def _despace_hyphens(text: str) -> str:
    """Hyphen -> space, preserving length so string indices stay valid."""
    return text.translate(_HYPHENS)


def _label_ends_cleanly(lower_line: str, after_idx: int) -> bool:
    """Decide whether a synonym match really ended where we think it did.

    A LETTER immediately after the synonym means we matched inside a longer word ("TV" inside
    "TVI") -- always rejected.

    A DIGIT is ambiguous, and both readings occur on real reports:
      "PW D0.8cm"  OCR dropped the space, so the digits are the VALUE  -> accept
      "LAd4 39.4"  the digit is part of the label's own name (LA diameter, 4-chamber), and
                   accepting it stores the ordinal "4" as the measurement -> reject
    They are told apart by what the digits look like: a real measurement has a decimal point or
    a unit attached, a label ordinal is a bare integer. Getting this wrong is not symmetric --
    accepting an ordinal writes a fictional value, so the ambiguous case fails closed.
    """
    if after_idx >= len(lower_line):
        return True
    nxt = lower_line[after_idx]
    if nxt.isalpha():
        return False
    if nxt.isdigit():
        return bool(_GLUED_MEASUREMENT_RE.match(lower_line, after_idx))
    return True


def _find_all_label_matches(line: str) -> List[Tuple[str, str, int, int]]:
    """
    Scan a line for ALL matching known labels (a single report line often
    holds two label:value pairs side by side, e.g. "MV: Normal   AV: Mild").
    Returns matches sorted by position, non-overlapping. Longest synonyms are
    tried first so short abbreviations (EF, MV, AV...) don't falsely match
    inside longer words/phrases, and don't get shadowed by a later, shorter
    synonym claiming the same span.
    """
    # Hyphens are normalized to spaces on BOTH sides so "wall-motion abnormalities" matches the
    # "Wall Motion" synonym, and "Sino-tubular Junction" still matches its own. The substitution
    # is one character for one character, so every index below still refers to the original
    # `line` -- which is what the caller slices the value out of.
    lower_line = _despace_hyphens(line.lower())
    occupied = [False] * len(line)
    found: List[Tuple[str, str, int, int]] = []

    for syn, canon in _SYNONYM_INDEX:
        syn_l = _despace_hyphens(syn.lower())
        search_from = 0
        while True:
            idx = lower_line.find(syn_l, search_from)
            if idx == -1:
                break
            after_idx = idx + len(syn_l)
            search_from = after_idx
            if any(occupied[idx:after_idx]):
                continue
            before_ok = idx == 0 or not lower_line[idx - 1].isalnum()
            after_ok = _label_ends_cleanly(lower_line, after_idx)
            if not (before_ok and after_ok):
                continue
            rem_after = lower_line[after_idx:].lstrip()
            # If the same canonical parameter has already been matched earlier on this line,
            # don't match a second synonym for it unless explicitly headed with a colon/equals
            if any(f[0] == canon for f in found) and not rem_after.startswith((":", "=")):
                continue
            # 'Aorta' in prose like 'No coarctation of aorta' is not an Ao_Diameter label
            if syn_l == "aorta" and re.search(r"\bcoarctation\b", lower_line):
                continue
            # If a short numeric abbreviation (like MVA, AVA) is followed by qualitative text and no colon, it's not a numeric label
            meta_k = PARAMETERS.get(canon, {}).get("kind")
            if meta_k == "numeric" and syn_l in ("mva", "ava", "edv", "esv", "sv", "co", "ci"):
                if not rem_after.startswith((":", "=", "-")) and not re.match(r"^\s*(\d+(?:\.\d+)?)", rem_after):
                    continue
            score = _fuzzy_ratio(syn_l, line[idx:after_idx])
            if score < LABEL_MATCH_THRESHOLD:
                continue
            for i in range(idx, after_idx):
                occupied[i] = True
            found.append((canon, syn, idx, after_idx))

    found.sort(key=lambda f: f[2])
    return found


def _extract_numeric_value(remainder: str) -> Optional[str]:
    match = find_measurement(remainder)
    if not match:
        return None
    number, unit = match.group(1), match.group(2)
    if unit:
        return f"{number}{unit if unit == '%' else ' ' + unit}".strip()
    return number


def _extract_text_value(remainder: str) -> Optional[str]:
    # Strip common separators (':' , '-', '=', '/') then take the qualitative phrase
    cleaned = re.sub(r"^[\s:.\-=/]+", "", remainder).strip()
    if not cleaned:
        return None
    # Stop at the next known label if two findings share a line (e.g. "MV: Normal   AV: Mild stenosis")
    matches = _find_all_label_matches(cleaned)
    for canon, syn, start, end in matches:
        if start == 0:
            continue
        rem_after = cleaned[end:].lstrip()
        if rem_after.startswith((":", "=", "-")):
            cleaned = cleaned[:start].strip(" \t:-=/")
            break
    return cleaned or None


def _extract_value_for_kind(kind: str, remainder: str) -> Optional[str]:
    if kind == "numeric":
        return _extract_numeric_value(remainder)
    if kind == "text":
        # The flat-line scan has no cell structure, so a short abbreviation can match inside an
        # unrelated row -- "TV" inside "TV Vmax 1.69 m/sec" stored a flow velocity as the
        # tricuspid valve FINDING on a real report. A qualitative field only ever holds
        # qualitative language, so require the remainder to actually contain some.
        value = _extract_text_value(remainder)
        return value if value and looks_qualitative(value) else None
    if kind == "chamber_size":
        # Try a raw measurement first (e.g. "4.4 cm"); if the report instead
        # gave a qualitative read (e.g. "Mildly Dilated"), fall back to that.
        numeric = _extract_numeric_value(remainder)
        if numeric is not None:
            return numeric
        return _extract_text_value(remainder)
    return _extract_text_value(remainder)


# ---------------------------------------------------------------------------
# Disqualifying-suffix check + whole-cell label matching (form fields / table cells)
# ---------------------------------------------------------------------------
def _is_disqualified(candidate_text: str, canon: str) -> bool:
    """A candidate label that carries a suffix token indicating a distinct (usually indexed)
    field must not be allowed to match the base field's synonym -- e.g. "LVEDD/BSA" must not
    satisfy "LVEDD" for the base LVIDd field. Indexed fields are exempt from their own suffix
    list (they're expected to carry these tokens)."""
    meta = PARAMETERS[canon]
    if meta.get("indexed_variant_of") is not None:
        return False
    suffixes = meta.get("disqualifying_suffixes") or ()
    norm = normalized_key(candidate_text)
    return any(suf in norm for suf in suffixes)


def _match_label_exact_or_fuzzy(candidate_text: str) -> Optional[Tuple[str, str, float]]:
    """Whole-string normalized match of a candidate label (a table cell or form-field key)
    against the synonym index. Stricter than the flat-line substring scan by construction
    (the whole cell must resemble the whole synonym), and additionally disqualifying-suffix
    checked. Returns (canonical, synonym, score) for the best match, or None."""
    norm_candidate = normalized_key(candidate_text)
    if not norm_candidate:
        return None

    best: Optional[Tuple[str, str, float]] = None
    for syn, canon in _SYNONYM_INDEX:
        norm_syn = normalized_key(syn)
        if not norm_syn:
            continue
        score = 100.0 if norm_candidate == norm_syn else _fuzzy_ratio(norm_syn, norm_candidate)
        if score < LABEL_MATCH_THRESHOLD:
            continue
        if _is_disqualified(candidate_text, canon):
            continue
        if best is None or score > best[2]:
            best = (canon, syn, score)
    return best


# ---------------------------------------------------------------------------
# Semantic (Groq, text-only) label resolution -- used ONLY for labels the
# deterministic dictionary matcher could not resolve.
# ---------------------------------------------------------------------------
def build_semantic_label_map(labels: List[str]) -> Dict[str, str]:
    """One batched Groq call per document mapping unresolved label variants -> canonical keys.

    Every returned key is re-validated here: it must be a real dictionary key, and it must still
    pass the disqualifying-suffix check. So even if the model insists "LVEDD/BSA" means "LVIDd",
    that answer is discarded rather than overwriting a primary measurement.

    Returns {normalized_label: canonical}. Empty on any failure -- Groq is an enhancement to
    dictionary matching here, never a dependency.
    """
    unresolved = [l for l in labels if l and _match_label_exact_or_fuzzy(l) is None]
    if not unresolved:
        return {}

    try:
        from app.ocr.groq_extractor import resolve_labels_semantically
        raw_map = resolve_labels_semantically(unresolved, list(PARAMETERS.keys()))
    except Exception:  # noqa: BLE001 -- includes GroqVisionError, import errors, anything else
        return {}

    validated: Dict[str, str] = {}
    for label, canon in raw_map.items():
        if canon not in PARAMETERS:
            continue
        if _is_disqualified(label, canon):
            continue
        validated[normalized_key(label)] = canon
    return validated


def _resolve_label(text: str, semantic_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Canonical key for a label cell, dictionary first, Groq's answer second."""
    match = _match_label_exact_or_fuzzy(text)
    if match:
        return match[0]
    if semantic_map:
        return semantic_map.get(normalized_key(text))
    return None


# ---------------------------------------------------------------------------
# Unit conversion + range checking (shared by every numeric extraction path)
# ---------------------------------------------------------------------------
def _convert_unit(number: float, from_unit: str, to_unit: str) -> float:
    if from_unit == to_unit:
        return number
    factor = _UNIT_CONVERSION_FACTORS.get((from_unit, to_unit))
    return number * factor if factor is not None else number


def _in_range(number: float, rng: Optional[Tuple[float, float]]) -> bool:
    if rng is None:
        return True
    lo, hi = rng
    return lo <= number <= hi


def _distance_outside(number: float, rng: Optional[Tuple[float, float]]) -> float:
    """How far `number` falls outside `rng` (0.0 when inside). Used only to choose between two
    candidate unit readings when neither is inside the range -- see the paediatric case in
    resolve_value's unit resolution."""
    if rng is None:
        return 0.0
    lo, hi = rng
    if number < lo:
        return lo - number
    if number > hi:
        return number - hi
    return 0.0


def _format_value_with_unit(number: float, canonical_unit: Optional[str]) -> str:
    display_unit = _UNIT_DISPLAY.get(canonical_unit, canonical_unit or "")
    if not display_unit:
        return f"{number:g}"
    if display_unit == "%":
        return f"{number:g}%"
    return f"{number:g} {display_unit}"


def _clean_valve_finding_text(text: str) -> Optional[str]:
    """Clean and normalize composite valve findings (e.g. 'Trileaflet , Degenerative' -> 'Trileaflet, Degenerative')."""
    if not text or not str(text).strip():
        return None
    cleaned = str(text)
    # Strip out parenthetical measurement expressions like (RVSP/TR: 18 mmHg) or (18 mmHg)
    cleaned = re.sub(r"\s*\(.*?\)", "", cleaned)
    # Strip unclosed parenthetical fragments at the end like (RVSP/TR:
    cleaned = re.sub(r"\s*\(.*$", "", cleaned)
    # If the text has non-valve notes like ", No PH", strip them out
    cleaned = re.sub(r",\s*no\s+ph\b.*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s\t]+", " ", cleaned).strip(" ,;:-(")
    low = cleaned.lower()
    # Check if this text contains recognized valve qualitative terms
    if not any(re.search(r"\b" + re.escape(term) + r"\b", low) for term in QUALITATIVE_TERMS):
        return None
    # If the text is just a bare single descriptor like 'normal' or 'trileaflet' or 'mild mr', let finding_phrase handle or clean it
    parts = [re.sub(r"\s+", " ", p.strip()) for p in cleaned.split(",") if p.strip()]
    if not parts:
        return cleaned.title()
    formatted = []
    for p in parts:
        words = []
        for w in p.split():
            w_low = w.lower()
            if w_low in ("mr", "tr", "ar", "pr", "ms", "as", "ts", "ps", "bav", "mvp", "ai", "ti", "pi", "mva", "ava"):
                words.append(w.upper())
            elif w_low in ("and", "with", "no", "non"):
                words.append(w.capitalize() if not words else w_low)
            else:
                words.append(w.capitalize())
        formatted.append(" ".join(words))
    return ", ".join(formatted)


def validate_and_map_qualitative(canon: str, raw_value_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Validate a qualitative finding against canonical clinical dropdown enums and vocabulary.

    Returns:
        (mapped_enum_value, flag_reason)
        If the value cannot be mapped to any valid clinical enum option, returns (None, error_reason).
    """
    if not raw_value_text or not str(raw_value_text).strip():
        return None, "Empty qualitative value"

    text = str(raw_value_text).strip()
    low = text.lower()
    meta = PARAMETERS.get(canon, {})
    kind = meta.get("kind")
    db_field = meta.get("db_field", "")

    # 1. Clots / Thrombus (DROPDOWN ENUM: "No Clots", "Clots Present")
    if canon == "Clots_Thrombus" or db_field == "clots_thrombus":
        if (low in ("no", "nil", "none", "absent", "negative", "normal", "wnl", "clear", "clean") or
                N.is_absent(text) or N.is_normal(text) or
                any(w in low for w in ("nil", "none", "absent", "not seen", "not detected",
                                       "not visualized", "not visualised", "no clot", "no thrombus",
                                       "no vegetation", "negative", "normal", "wnl", "free of",
                                       "clean", "clear", "unremarkable", "no mass", "without"))):
            return "No Clots", None
        if any(w in low for w in ("present", "detected", "seen", "positive", "clot", "thrombus",
                                  "thrombi", "vegetation", "vegetations", "mass", "masses")):
            return "Clots Present", None
        return None, f"Unrecognized finding '{text}' for Clots/Thrombus"

    # 2. Pericardial Effusion (DROPDOWN ENUM: "None/Trace", "Small", "Moderate", "Large")
    if canon == "Pericardial_Effusion" or db_field == "pericardial_effusion":
        if (low in ("no", "nil", "none", "absent", "negative", "normal", "wnl") or
                N.is_absent(text) or N.is_normal(text) or
                any(w in low for w in ("none", "no effusion", "absent", "nil", "not seen",
                                       "not detected", "not visualized", "not visualised",
                                       "normal", "unremarkable", "none/trace", "trace", "trivial",
                                       "physiological", "no fluid", "free of effusion", "negative",
                                       "wnl", "no pericardial effusion"))):
            return "None/Trace", None
        if any(w in low for w in ("large", "severe", "gross", "massive", "tamponade")) or "> 20" in low or "> 2" in low:
            return "Large", None
        if any(w in low for w in ("moderate", "mod", "circumferential moderate")) or "10-20" in low or "1-2" in low:
            return "Moderate", None
        if any(w in low for w in ("small", "mild", "minimal")) or "< 10" in low or "< 1" in low or any(w in low for w in ("effusion", "present", "fluid")):
            return "Small", None
        return None, f"Unrecognized finding '{text}' for Pericardial Effusion"

    # 3. RWMA (DROPDOWN ENUM: "Absent", "Present")
    if canon == "RWMA" or db_field == "rwma":
        if N.is_absent(text) or N.is_normal(text) or any(w in low for w in ("absent", "no rwma", "nil", "none", "not seen", "negative", "no regional")):
            return "Absent", None
        if any(w in low for w in ("present", "detected", "seen", "positive", "yes", "hypokinetic", "akinetic", "dyskinetic", "rwma present")):
            return "Present", None
        return None, f"Unrecognized finding '{text}' for RWMA"

    # 4. Wall Motion & Segmentals (DROPDOWN ENUM: "Normal", "Hypokinetic", "Akinetic", "Dyskinetic")
    if canon in ("Wall_Motion", "Septal_Wall_Motion", "Anterior_Wall_Motion", "Inferior_Wall_Motion",
                 "Lateral_Wall_Motion", "Posterior_Wall_Motion", "Apical_Wall_Motion") or (db_field and "wall_motion" in db_field):
        if any(w in low for w in ("dyskinetic", "dyskinesia", "dyskinesis", "paradoxical", "aneurysm", "aneurysmal")):
            return "Dyskinetic", None
        if any(w in low for w in ("akinetic", "akinesia", "akinesis", "no motion", "scar")):
            return "Akinetic", None
        if any(w in low for w in ("hypokinetic", "hypokinesia", "hypokinesis", "depressed", "sluggish")):
            return "Hypokinetic", None
        if N.is_normal(text) or any(w in low for w in ("normal", "normokinetic", "good", "preserved", "intact", "unremarkable", "wnl", "no hypokinesia")):
            return "Normal", None
        return None, f"Unrecognized wall motion finding '{text}'"

    # 5. IVC (DROPDOWN ENUM: "Normal", "Dilated/Plethoric")
    if canon == "IVC" or db_field == "ivc":
        if any(w in low for w in ("dilated", "plethoric", "engorged", "non-collapsing", "non collapsible", "< 50", "<50")):
            return "Dilated/Plethoric", None
        if N.is_normal(text) or any(w in low for w in ("normal", "not dilated", "> 50", ">50", "collapsing", "collapsible", "respiration", "respiratory")):
            return "Normal", None
        if _has_number(text):
            return text, None
        if looks_qualitative(text):
            return text.title(), None
        return None, f"Unrecognized IVC finding '{text}'"

    # 6. Chamber size qualitative fields (ra_size, rv_size, lvidd, lvids)
    if kind == "chamber_size":
        if _has_number(text):
            return text, None
        if looks_qualitative(text):
            return text.title(), None
        return None, f"Unrecognized chamber size descriptor '{text}'"

    # 7. Valves (MV, AV, TV, PV)
    if canon in ("MV", "AV", "TV", "PV") or (db_field and db_field.endswith("_finding")):
        cleaned_valve = _clean_valve_finding_text(text)
        if cleaned_valve:
            return cleaned_valve, None
        p = finding_phrase(text, canon=canon)
        if p:
            return p, None
        if looks_qualitative(text):
            return text.title(), None
        return None, f"Unrecognized valve finding '{text}'"

    # 8. Numeric fields receiving qualitative values (EF, IVSd, PWd, PASP, LVOT_Peak_Gradient, etc.)
    if kind == "numeric":
        if _has_number(text):
            return text, None
        if any(w in low for w in ("severe", "severely depressed", "severely reduced", "poor")):
            return "Severe", "Report gave qualitative grade ('Severe') instead of numeric measurement"
        if any(w in low for w in ("moderate", "moderately depressed", "moderately reduced")):
            return "Moderate", "Report gave qualitative grade ('Moderate') instead of numeric measurement"
        if any(w in low for w in ("mild", "mildly depressed", "mildly reduced")):
            return "Mild", "Report gave qualitative grade ('Mild') instead of numeric measurement"
        if N.is_normal(text) or any(w in low for w in ("normal", "good", "preserved", "intact", "wnl", "unremarkable")):
            return "Normal", "Report gave qualitative descriptor ('Normal') instead of numeric measurement"
        return None, f"Value '{text}' is not a numeric measurement or recognized clinical grade"

    # Default qualitative
    if looks_qualitative(text):
        return text, None
    return None, f"Unrecognized qualitative value '{text}'"


def resolve_value(
    canon: str, raw_value_text: Optional[str], raw_confidence: float, source: str,
    source_snippet: str, value_type: Optional[str] = None,
) -> ExtractedField:
    """
    Single centralized function every match_* path (form_field / table_grid / table /
    flat_line / narrative) funnels through. In order:
      1. Confidence reject gate (checked BEFORE any value parsing -- a low-confidence region
         must never be handed to anything that could guess a value for it).
      2. Classify the value as numeric or qualitative.
      3. QUALITATIVE -> validate against canonical dropdown enums and store mapped value.
      4. NUMERIC -> regex-extract (raw_number, detected_unit).
      5. Convert to the field's canonical_unit -- explicit unit if given; if ambiguous (no
         unit symbol), infer from which interpretation is physiologically plausible.
      6. Hard-reject (store nothing) if the final canonical-unit value falls outside
         impossible_range -- catches OCR digit errors like a dropped decimal point.
      7. Soft-flag (store, but flag for review) if outside valid_range.
      8. Format back into the SAME "<number> <unit>" string convention already used in the
         DB columns, so app/predictor/normalize.py keeps working unmodified.

    value_type may be supplied by the caller (e.g. Groq said this narrative finding is
    qualitative); it is otherwise inferred. A caller-supplied type is only honoured when the
    text supports it, so an upstream mistake cannot force a word down the numeric path.
    """
    meta = PARAMETERS[canon]
    kind = meta["kind"]
    db_field = meta["db_field"]
    conf = round(float(raw_confidence), 2)
    snippet = (source_snippet or "").strip()

    result_meta = {
        "source": source,
        "source_snippet": snippet,
        "value_type": None,
        "raw_detected_value": None,
        "raw_detected_unit": None,
        "conversion_applied": False,
        "final_stored_value": None,
        "confidence": conf,
        "flagged": False,
        "flag_reason": None,
    }

    def _reject(reason: str) -> ExtractedField:
        result_meta["flagged"] = True
        result_meta["flag_reason"] = reason
        return ExtractedField(canon, db_field, None, conf, snippet, None, True, source, None,
                              result_meta)

    def _store_qualitative(text: str, extra_reason: Optional[str] = None) -> ExtractedField:
        # Unit/conversion metadata is REMOVED rather than set to None: those keys are
        # meaningless for a qualitative finding and their presence would imply otherwise.
        for numeric_only_key in ("raw_detected_value", "raw_detected_unit", "conversion_applied"):
            result_meta.pop(numeric_only_key, None)
        cleaned_text = (text or "").strip(" \t:-=.,")
        result_meta["value_type"] = VALUE_TYPE_QUALITATIVE
        result_meta["final_stored_value"] = cleaned_text
        reason = extra_reason
        if conf < OCR_CONFIDENCE_FLAG_THRESHOLD:
            reason = reason or "Low OCR confidence -- please confirm against original report."
        result_meta["flagged"] = bool(reason)
        result_meta["flag_reason"] = reason
        return ExtractedField(canon, db_field, cleaned_text, conf, snippet, None, bool(reason), source,
                              VALUE_TYPE_QUALITATIVE, result_meta)

    # 1. Confidence gate -- before ANY value extraction is attempted.
    if conf < OCR_CONFIDENCE_REJECT_THRESHOLD:
        return _reject(
            "Source region unreadable/likely blank -- very low OCR confidence, not extracted."
        )

    if raw_value_text is None or not str(raw_value_text).strip():
        return _reject("No value text found for this parameter.")
    raw_value_text = str(raw_value_text).strip()

    # Segmental measurement detection for wall thickness parameters (IVSd / PWd / IVSs / PWs)
    seg_info = None
    if canon in ("IVSd", "PWd", "IVSs", "PWs"):
        seg_info = extract_segmental_measurements(raw_value_text) or extract_segmental_measurements(snippet)
        if seg_info:
            result_meta["segmental"] = True
            result_meta["segments"] = seg_info["segments"]
            result_meta["segment_values_mm"] = seg_info["segment_values_mm"]
            result_meta[f"{db_field}_basal"] = seg_info["segment_values_mm"].get("basal")
            result_meta[f"{db_field}_mid"] = seg_info["segment_values_mm"].get("mid")
            result_meta[f"{db_field}_apical"] = seg_info["segment_values_mm"].get("apical")
            raw_value_text = seg_info["basal_formatted"]

    # 2. Classify. A caller-supplied "numeric" is downgraded if there is no number to read.
    inferred = classify_value_type(raw_value_text, kind)
    if value_type == VALUE_TYPE_QUALITATIVE and inferred is not None:
        inferred = VALUE_TYPE_QUALITATIVE

    if inferred is None:
        return _reject("No usable numeric or qualitative value could be read from the source text.")

    # 3. Qualitative -> validate against canonical dropdown enums and store mapped value.
    if inferred == VALUE_TYPE_QUALITATIVE:
        mapped_val, map_reason = validate_and_map_qualitative(canon, raw_value_text)
        if mapped_val is None:
            return _reject(map_reason or f"Unrecognized qualitative value '{raw_value_text}' for {canon}.")
        extra = map_reason
        if kind == "numeric" and not extra:
            extra = ("Report gave a descriptive finding where a measurement is usually printed "
                     "-- please confirm against original report.")
        return _store_qualitative(mapped_val, extra)

    # 4. Numeric: pull the number out. classify_value_type guaranteed SOME number is present,
    #    but every one of them may be a reference-range bound -- in which case this field has no
    #    patient value and must be rejected rather than given the range's lower bound.
    num_match = find_measurement(raw_value_text)
    if num_match is None:
        return _reject("Only a reference/normal range was printed here, not a measured value.")
    result_meta["value_type"] = VALUE_TYPE_NUMERIC
    raw_number_str = num_match.group(1)
    detected_unit = normalize_unit(num_match.group(2))
    result_meta["raw_detected_value"] = raw_number_str
    result_meta["raw_detected_unit"] = detected_unit

    try:
        number = float(raw_number_str)
    except ValueError:
        return _reject("Value could not be parsed as a number.")

    canonical_unit = meta["canonical_unit"]
    alternate_unit = meta["alternate_unit"]
    valid_range = meta["valid_range"]
    impossible_range = meta["impossible_range"]
    conversion_applied = False
    unit_inferred_flag = None

    # 5. Unit resolution. Never strip a unit silently: the detected unit is recorded above, the
    #    conversion is explicit, and both raw and converted values are kept for audit.
    if detected_unit == canonical_unit or (not canonical_unit and detected_unit is None):
        final_number = number
    elif alternate_unit and detected_unit == alternate_unit:
        final_number = _convert_unit(number, alternate_unit, canonical_unit)
        conversion_applied = True
    elif detected_unit is None and alternate_unit:
        canonical_plausible = _in_range(number, valid_range)
        converted = _convert_unit(number, alternate_unit, canonical_unit)
        alternate_plausible = _in_range(converted, valid_range)
        prefer_converted = alternate_plausible and not canonical_plausible
        if not prefer_converted and not canonical_plausible and not alternate_plausible:
            # NEITHER reading lands in the adult reference window. That is exactly what a
            # paediatric report looks like: "LVPWS -5" in a block headed "LV STUDY(MM)" is
            # 5 mm = 0.5 cm, which sits just under PWs' 0.6 cm floor, so the plausibility test
            # above rejects both readings and 5 is stored as 5 CENTIMETRES -- a whole order of
            # magnitude wrong, on a child. Fall back to whichever reading misses the window by
            # less; 0.5 is 0.1 below it, 5 is 2.0 above it.
            prefer_converted = _distance_outside(converted, valid_range) < _distance_outside(
                number, valid_range
            )
        if prefer_converted:
            final_number = converted
            conversion_applied = True
            unit_inferred_flag = (
                "Unit inferred from value magnitude (no unit symbol detected in source) -- "
                "please confirm against original report."
            )
        else:
            final_number = number
    else:
        # A unit token was present but matches neither canonical nor alternate -- can't be
        # trusted; treat the bare number as already-canonical and let range checks catch it.
        final_number = number

    result_meta["conversion_applied"] = conversion_applied

    # 6. Hard sanity bound -- reject, don't store.
    # SAFETY INVARIANT: everything from here down is numeric-only. A qualitative value already
    # returned at step 3, so it can never reach these bounds checks.
    assert result_meta["value_type"] == VALUE_TYPE_NUMERIC
    if impossible_range and not _in_range(final_number, impossible_range):
        shown = _format_value_with_unit(final_number, canonical_unit)
        result_meta["flagged"] = True
        result_meta["flag_reason"] = (
            f"Value outside physiologically possible range ({shown}) -- likely OCR error such as "
            f"a missed decimal point or misread digit. Original source text: '{snippet}'"
        )
        result_meta["final_stored_value"] = None
        return ExtractedField(canon, db_field, None, conf, snippet, None, True, source,
                              VALUE_TYPE_NUMERIC, result_meta)

    # 7. Soft bound -- store, flag for doctor review (may be a genuine abnormal finding).
    in_range = _in_range(final_number, valid_range) if valid_range else None
    flagged = False
    flag_reason = None
    if valid_range and not in_range:
        flagged = True
        flag_reason = "Value outside normal clinical range -- flagged for doctor review."
    if unit_inferred_flag:
        flagged = True
        flag_reason = unit_inferred_flag
    if conf < OCR_CONFIDENCE_FLAG_THRESHOLD:
        flagged = True
        flag_reason = flag_reason or "Low OCR confidence -- please confirm against original report."

    # 8. Format back into the DB column's existing string convention.
    final_value_str = _format_value_with_unit(final_number, canonical_unit)
    result_meta["final_stored_value"] = final_value_str
    result_meta["flagged"] = flagged
    result_meta["flag_reason"] = flag_reason

    return ExtractedField(canon, db_field, final_value_str, conf, snippet, in_range, flagged,
                          source, VALUE_TYPE_NUMERIC, result_meta)


# ---------------------------------------------------------------------------
# Priority (a): form fields
# ---------------------------------------------------------------------------
def match_form_fields(
    form_fields: List[dict], semantic_map: Optional[Dict[str, str]] = None
) -> Dict[str, ExtractedField]:
    results: Dict[str, ExtractedField] = {}
    for ff in form_fields:
        canon = _resolve_label(ff.get("key_text", ""), semantic_map)
        if not canon:
            continue
        if canon in results and results[canon].value is not None:
            continue
        snippet = f'{ff.get("key_text", "")}: {ff.get("value_text", "")}'
        ef = resolve_value(canon, ff.get("value_text"), ff.get("value_confidence", 0.0), "form_field", snippet)
        # Carry through any cross-validation evidence (confidence_basis, corroborated,
        # conflicting_value...) so extraction_meta records HOW this confidence was arrived at
        # and never implies a Groq-sourced number came from a real OCR score.
        extra = ff.get("extra_meta") or {}
        if extra:
            ef.meta.update(extra)
            if extra.get("conflict_warning"):
                ef.flagged = True
                ef.meta["flagged"] = True
                ef.meta["flag_reason"] = extra["conflict_warning"]
        results[canon] = ef
    return results


# ---------------------------------------------------------------------------
# Priority (b): tables -- read across a row only, never down to another row
# ---------------------------------------------------------------------------
def _scan_row_for_label_value_pairs(
    cells: List[dict], semantic_map: Optional[Dict[str, str]] = None
) -> List[Tuple[str, str, dict, dict]]:
    """Walk a row's cells left to right. A label match at index i takes cells[i+1] as its
    value -- whatever comes immediately after wherever the label was found in THIS row, which
    naturally handles a dense multi-column row (label|value|label|value) as multiple
    independent pairs, and never looks past this row's own cells."""
    pairs: List[Tuple[str, str, dict, dict]] = []
    used = [False] * len(cells)
    i = 0
    while i < len(cells):
        if used[i]:
            i += 1
            continue
        canon = _resolve_label(cells[i].get("text", ""), semantic_map)
        if canon and i + 1 < len(cells) and not used[i + 1]:
            pairs.append((canon, cells[i].get("text", ""), cells[i], cells[i + 1]))
            used[i] = True
            used[i + 1] = True
            i += 2
        else:
            i += 1
    return pairs


def match_tables(
    tables: List[dict], semantic_map: Optional[Dict[str, str]] = None
) -> Dict[str, ExtractedField]:
    results: Dict[str, ExtractedField] = {}
    for table in tables:
        for row in table.get("rows", []):
            pairs = _scan_row_for_label_value_pairs(row.get("cells", []), semantic_map)
            for canon, syn, label_cell, value_cell in pairs:
                if canon in results and results[canon].value is not None:
                    continue
                snippet = f'{label_cell.get("text", "")} | {value_cell.get("text", "")}'
                ef = resolve_value(canon, value_cell.get("text"), value_cell.get("confidence", 0.0), "table", snippet)
                results[canon] = ef
    return results


# ---------------------------------------------------------------------------
# Priority (b): table_grid -- Stage 1's geometrically reconstructed grid
# ---------------------------------------------------------------------------
def _is_value_like(text: str) -> bool:
    """A cell that reads as a measurement or a finding, i.e. a value rather than a label."""
    return _has_number(text) or looks_qualitative(text)


# How many columns to the right of a label its value may sit. 2 covers "label | value" and the
# over-split "label | (fragment) | value"; beyond that the cell belongs to another section.
_MAX_VALUE_COLUMN_DISTANCE = 2


def _split_merged_cell(
    cell: dict, semantic_map: Optional[Dict[str, str]] = None
) -> Optional[Tuple[str, dict, dict]]:
    """Recover a (label, value) pair from ONE cell that contains both.

    Stage 1 keeps label and value in separate cells whenever the geometry allows, but OCR
    sometimes emits them as a single detection ("RAD (major3.3") or the columns are too tight
    to separate. Such a cell is rejected as a label (it holds a number) and is therefore
    invisible to the normal path -- the parameter is lost outright.

    Reuses the flat-line machinery rather than a second matcher: _find_all_label_matches already
    does longest-synonym-first matching with boundary guards, and the value is the text after
    the label, exactly as match_flat_lines slices it. Confidence is the cell's own -- a split
    changes where we read the value from, never how much we trust it.
    """
    text = (cell.get("text") or "").strip()
    matches = _find_all_label_matches(text)
    if not matches:
        return None

    canon, _syn, _start, end = matches[0]
    # Stop at the next label so "IVSd 1.0 PWd 0.8" cannot give IVSd the value 0.8.
    next_start = matches[1][2] if len(matches) > 1 else len(text)
    remainder = text[end:next_start]
    if not remainder.strip():
        return None

    label_cell = dict(cell, text=text[:end])
    value_cell = dict(cell, text=remainder)
    return (canon, label_cell, value_cell)


def _scan_grid_row(
    cells: List[dict], semantic_map: Optional[Dict[str, str]] = None
) -> List[Tuple[str, dict, dict]]:
    """Resolve one grid row into (canonical, label_cell, value_cell) triples.

    The label is column 0 when column 0 is label-like; otherwise the first cell in the row that
    is neither numeric nor a qualitative term. A label's candidate values are ONLY the cells to
    its right, up to the next label in the SAME row -- never a cell from another row, and never
    a cell to the left of the label. That is the structural guarantee: the semantic layer is
    physically never shown content from the wrong row or the wrong column, so an off-by-one row
    shift or a label/value column swap has no way to occur.

    Rows laid out label|value|label|value (the dense multi-column report case) resolve as
    multiple independent pairs by the same rule.
    """
    labels: List[Tuple[int, str]] = []
    merged_pairs: List[Tuple[str, dict, dict]] = []
    for i, cell in enumerate(cells):
        text = (cell.get("text") or "").strip()
        # A cell holding a number is a value, never a label. Note the test is _has_number rather
        # than _is_value_like: several real LABELS are themselves qualitative words
        # ("Pericardial Effusion", "Clots / Thrombus"), so screening on the qualitative
        # vocabulary here would discard them. Qualitative VALUES are safe either way -- "Normal",
        # "Mild Regurgitation" and friends resolve to no canonical key at all.
        if not text:
            continue
        if _has_number(text):
            # A cell holding BOTH a label and a number is a merged cell -- OCR lost the column
            # boundary ("RAD (major3.3", "EF (Teich) 73.3%"). Splitting it recovers the pair;
            # without this the cell matches nothing at all and the parameter is simply lost.
            split = _split_merged_cell(cell, semantic_map)
            if split is not None:
                merged_pairs.append(split)
            continue
        canon = _resolve_label(text, semantic_map)
        if canon:
            labels.append((i, canon))

    pairs: List[Tuple[str, dict, dict]] = list(merged_pairs)
    for pos, (idx, canon) in enumerate(labels):
        end = labels[pos + 1][0] if pos + 1 < len(labels) else len(cells)
        # Bound how far right a value may sit. On templates with interleaved Normal(F/M)
        # columns, a label whose own value cell is blank otherwise reaches across the row into
        # a different SECTION -- "IV.Right Atrium" took "Lateral e'" from the Diastolic
        # Function block three columns away. A real value is in the next column, or the one
        # after it when Stage 1 over-split; anything further belongs to something else.
        end = min(end, idx + 1 + _MAX_VALUE_COLUMN_DISTANCE)
        span = cells[idx + 1:end]
        value_cell = next((c for c in span if (c.get("text") or "").strip()), None)
        if value_cell is None:
            if not span:
                continue
            # The label is printed but its value cell is empty -- either genuinely blank on the
            # report, or filtered out by Stage 1's confidence gate. Emit the empty cell anyway
            # so resolve_value records an explicit flagged null. Silently skipping would render
            # as "Not detected" with no flag, which in a clinical UI reads as "the report does
            # not contain this measurement" rather than "we could not read it".
            value_cell = span[0]
        pairs.append((canon, cells[idx], value_cell))
    return pairs


def match_table_grid(
    table_grid: List[List[dict]], semantic_map: Optional[Dict[str, str]] = None
) -> Dict[str, ExtractedField]:
    """Match parameters against Stage 1's reconstructed (row, column) grid."""
    results: Dict[str, ExtractedField] = {}
    for row in table_grid or []:
        for canon, label_cell, value_cell in _scan_grid_row(row, semantic_map):
            if canon in results and results[canon].value is not None:
                continue
            snippet = f'{label_cell.get("text", "")} | {value_cell.get("text", "")}'
            results[canon] = resolve_value(
                canon, value_cell.get("text"), value_cell.get("confidence", 0.0), "table", snippet
            )
    return results


# ---------------------------------------------------------------------------
# Priority (b.2): Multi-column 2D Doppler Measurement Tables
# ---------------------------------------------------------------------------
_VALVE_COL_PATTERNS = [
    (re.compile(r"\b(?:mitral|mv)\b", re.I), "MV"),
    (re.compile(r"\b(?:aortic|av)\b", re.I), "AV"),
    (re.compile(r"\b(?:tricuspid|tv)\b", re.I), "TV"),
    (re.compile(r"\b(?:pulmonic|pulmonary|pv)\b", re.I), "PV"),
]

_ROW_TYPE_PATTERNS = [
    (re.compile(r"^\s*(?:peak\s+gradient|peak\s+pg|max\s+pg|pg\s*max|max\s+gradient|peak\s+grad)\b", re.I), "Peak_Gradient"),
    (re.compile(r"^\s*(?:mean\s+gradient|mean\s+pg|mg\b|mean\s+grad)\b", re.I), "Mean_Gradient"),
    (re.compile(r"^\s*(?:peak\s+velocity|peak\s+vel|vmax|max\s+vel|velocity|max\s+velocity)\b", re.I), "Peak_Velocity"),
    (re.compile(r"^\s*(?:valve\s+area|mva|ava|tva|pva|area\b)", re.I), "Area"),
    (re.compile(r"^\s*(?:grade\s+of\s+regurgitation|regurgitation\s+grade|regurgitation|grade\b)", re.I), "Finding"),
]

_EMPTY_CELL_MARKERS = {"-", "--", "---", "nil", "none", "n/a", "na", ".", "..", "...", "", "–", "—", "/"}


def extract_doppler_matrix(doc_result: dict) -> Dict[str, ExtractedField]:
    """Parse 2D multi-column Doppler measurement matrix tables across columns (Mitral/Aortic/Tricuspid/Pulmonary).
    
    Handles both:
    1. Reconstructed table grids and digital tables (table_grid, tables)
    2. Formatted text lines (lines)
    """
    results: Dict[str, ExtractedField] = {}
    lines = doc_result.get("lines") or []
    tables = doc_result.get("tables") or []
    table_grid = doc_result.get("table_grid") or []

    # 1. From structured table_grid and tables
    all_table_rows_list = []
    if table_grid:
        all_table_rows_list.append(table_grid)
    for t in tables:
        rows = t.get("rows") or []
        row_cells = [r.get("cells", []) for r in rows]
        if row_cells:
            all_table_rows_list.append(row_cells)

    for grid in all_table_rows_list:
        col_map = {}
        for row in grid:
            cell_texts = [(c.get("text") or "").strip() for c in row]
            # Check if this row is a valve header row
            valve_headers = {}
            for col_idx, txt in enumerate(cell_texts):
                for pat, valve in _VALVE_COL_PATTERNS:
                    if pat.search(txt):
                        valve_headers[col_idx] = valve
                        break
            if len(set(valve_headers.values())) >= 2:
                col_map = valve_headers
                continue

            if col_map and cell_texts:
                label_text = cell_texts[0]
                matched_type = None
                for pat, r_type in _ROW_TYPE_PATTERNS:
                    if pat.search(label_text):
                        matched_type = r_type
                        break
                if not matched_type:
                    continue

                for col_idx, valve in col_map.items():
                    if col_idx < len(row):
                        cell = row[col_idx]
                        raw_text = (cell.get("text") or "").strip()
                        if not raw_text or raw_text.lower() in _EMPTY_CELL_MARKERS:
                            continue
                        canon = f"{valve}_{matched_type}" if matched_type != "Finding" else valve
                        if canon in PARAMETERS:
                            conf = cell.get("confidence", 95.0)
                            results[canon] = resolve_value(
                                canon, raw_text, conf, "doppler_table", f"{label_text} | {valve} = {raw_text}"
                            )

    # 2. From formatted lines (flat lines)
    lines_text = [l for l, c in lines]
    lines_conf = [c for l, c in lines]

    header_idx = None
    header_cols = []
    for i, line in enumerate(lines_text):
        cols = []
        for m_pat, valve in _VALVE_COL_PATTERNS:
            for m in m_pat.finditer(line):
                cols.append((m.start(), m.end(), valve))
        cols.sort(key=lambda x: x[0])
        distinct_valves = set(v for _, _, v in cols)
        if len(distinct_valves) >= 2:
            header_idx = i
            header_cols = cols
            break

    if header_idx is not None and header_cols:
        num_cols = len(header_cols)
        col_bounds = []
        for idx in range(num_cols):
            left = max(0, header_cols[idx][0] - 6) if idx == 0 else (header_cols[idx - 1][1] + header_cols[idx][0]) // 2
            right = len(lines_text[header_idx]) + 50 if idx == num_cols - 1 else (header_cols[idx][1] + header_cols[idx + 1][0]) // 2
            col_bounds.append((left, right, header_cols[idx][2]))

        for line_idx in range(header_idx + 1, min(header_idx + 15, len(lines_text))):
            line = lines_text[line_idx]
            conf = lines_conf[line_idx]
            if not line.strip():
                continue
            if re.match(r"^\s*(?:impression|conclusion|findings|summary|comment|echo)\b", line, re.I):
                break

            matched_type = None
            row_label_end = 0
            for r_pat, r_type in _ROW_TYPE_PATTERNS:
                m = r_pat.search(line)
                if m:
                    matched_type = r_type
                    rem_span = line[m.end():]
                    unit_m = re.match(r"^\s*(?:[^\w\s-]|–|-|:|\([^)]*\))*\s*", rem_span)
                    row_label_end = m.end() + (unit_m.end() if unit_m else 0)
                    break
            if not matched_type:
                continue

            remainder = line[row_label_end:].strip()
            raw_tokens = [re.sub(r"\s+", " ", t.strip()) for t in re.split(r"\s{3,}|\t+", remainder) if t.strip()]

            valve_regurg_matches = {}
            if matched_type == "Finding":
                regurg_patterns = {
                    "MV": r"\b((?:no|nil|trace|trivial|mild|mod|moderate|severe|sev)\s+mr|no\s+regurgitation|mild\s+regurgitation|moderate\s+regurgitation|severe\s+regurgitation|-)\b",
                    "AV": r"\b((?:no|nil|trace|trivial|mild|mod|moderate|severe|sev)\s+ar|no\s+regurgitation|mild\s+regurgitation|moderate\s+regurgitation|severe\s+regurgitation|-)\b",
                    "TV": r"\b((?:no|nil|trace|trivial|mild|mod|moderate|severe|sev)\s+tr|no\s+regurgitation|mild\s+regurgitation|moderate\s+regurgitation|severe\s+regurgitation|-)\b",
                    "PV": r"\b((?:no|nil|trace|trivial|mild|mod|moderate|severe|sev)\s+pr|no\s+regurgitation|mild\s+regurgitation|moderate\s+regurgitation|severe\s+regurgitation|-)\b",
                }
                for _, _, v in col_bounds:
                    pat = regurg_patterns.get(v)
                    if pat:
                        m_v = re.search(pat, remainder, re.IGNORECASE)
                        if m_v:
                            valve_regurg_matches[v] = m_v.group(1).strip()

            for col_idx, (left, right, valve) in enumerate(col_bounds):
                val_text = None
                if matched_type == "Finding" and valve in valve_regurg_matches:
                    val_text = valve_regurg_matches[valve]
                elif len(raw_tokens) == num_cols:
                    val_text = raw_tokens[col_idx]
                else:
                    slice_left = max(left, row_label_end)
                    slice_right = min(right, len(line))
                    if slice_left < slice_right:
                        val_text = line[slice_left:slice_right].strip(" \t:.,–—")

                if not val_text or val_text.lower() in _EMPTY_CELL_MARKERS:
                    continue

                canon = f"{valve}_{matched_type}" if matched_type != "Finding" else valve
                if canon in PARAMETERS and (canon not in results or results[canon].value is None):
                    results[canon] = resolve_value(
                        canon, val_text, conf, "doppler_table", f"{valve} {matched_type}: {val_text}"
                    )

    return results


# ---------------------------------------------------------------------------
# Priority (c): flat-line fuzzy fallback (last resort)
# ---------------------------------------------------------------------------
def match_flat_lines(ocr_lines: List[Tuple[str, float]]) -> Dict[str, ExtractedField]:
    results: Dict[str, ExtractedField] = {}

    for i, (line, conf) in enumerate(ocr_lines):
        matches = _find_all_label_matches(line)
        if not matches:
            continue

        for m_idx, (canon, syn, start, end) in enumerate(matches):
            meta = PARAMETERS[canon]
            next_start = matches[m_idx + 1][2] if m_idx + 1 < len(matches) else len(line)
            prev_end = matches[m_idx - 1][3] if m_idx > 0 else 0
            remainder = line[end:next_start]
            left_span = line[prev_end:start].strip(" \t:,-=")
            line_conf = conf

            value = None
            line_has_leading_negation = bool(re.match(r"^\s*(?:no|nil|none|without|free of|negative)\b", line, re.IGNORECASE))
            if line_has_leading_negation and meta["kind"] == "text":
                value = f"No {syn}"
            # Valve shorthand tokens (MR, AR, TR, PR, MS, AS, TS, PS) often have their modifier to the left (e.g. "Mild MR", "No AR")
            elif syn.lower() in ("mr", "ar", "tr", "pr", "ms", "as", "ts", "ps"):
                has_right_separator = bool(re.match(r"^\s*[:\-=]", remainder))
                if has_right_separator:
                    right_val = _extract_value_for_kind(meta["kind"], remainder)
                    if right_val:
                        value = finding_phrase(f"{right_val} {syn}", canon=canon) or (f"No {syn.upper()}" if _NEGATION_RE.search(right_val) else f"{right_val} {syn.upper()}")
                if not value and left_span and (looks_qualitative(left_span) or _NEGATION_RE.search(left_span)):
                    value = finding_phrase(f"{left_span} {syn}", canon=canon) or f"{left_span} {syn.upper()}"
                if not value:
                    right_val = _extract_value_for_kind(meta["kind"], remainder)
                    if right_val:
                        value = finding_phrase(f"{right_val} {syn}", canon=canon) or (f"No {syn.upper()}" if _NEGATION_RE.search(right_val) else f"{right_val} {syn.upper()}")
            else:
                value = _extract_value_for_kind(meta["kind"], remainder)
                is_dedicated_valve_label = (
                    canon in ("AV", "MV", "TV", "PV")
                    and (
                        syn.lower() in ("aortic valve", "mitral valve", "tricuspid valve", "pulmonary valve")
                        or bool(re.match(r"^\s*(?:aortic|mitral|tricuspid|pulmonary)\s+valve\b", line, re.IGNORECASE))
                    )
                )
                if is_dedicated_valve_label:
                    full_valve_text = remainder
                    k = i + 1
                    while k < len(ocr_lines):
                        next_l, next_c = ocr_lines[k]
                        next_strip = next_l.strip()
                        if not next_strip:
                            k += 1
                            continue
                        if _is_impression_heading(next_l) or _line_is_heading(next_l) or _starts_a_new_section(next_l) or _find_all_label_matches(next_l):
                            break
                        if looks_qualitative(next_strip) or _NEGATION_RE.search(next_strip) or any(w in next_strip.lower() for w in ("as", "ar", "mr", "tr", "pr", "ms", "ts", "ps", "no ", "mild", "mod", "severe", "trileaflet", "bicuspid", "degenerative", "calcif", "sclerot", "prolapse", "adequate", "thickened")):
                            full_valve_text = (full_valve_text + " , " + next_strip).strip(" ,")
                            k += 1
                        else:
                            break
                    val_phrase = _clean_valve_finding_text(full_valve_text) or finding_phrase(full_valve_text, canon=canon)
                    if val_phrase:
                        value = val_phrase
                elif value is None and meta["kind"] == "text":
                    if left_span and (looks_qualitative(left_span) or _NEGATION_RE.search(left_span)):
                        value = finding_phrase(left_span, canon=canon) or left_span
                elif value is not None and meta["kind"] == "text":
                    val_phrase = finding_phrase(value, canon=canon) or _clean_valve_finding_text(value)
                    if val_phrase:
                        value = val_phrase
                    elif "regurgitation" in syn.lower() and "regurgitation" not in value.lower() and not N.contains_word(value, "mr", "ar", "tr", "pr"):
                        val_phrase = finding_phrase(f"{value} {syn}", canon=canon)
                        if val_phrase:
                            value = val_phrase

            # Label-only line (e.g. "LEFT ATRIUM" as a section header, value printed on the
            # next line) -> look one line ahead for the value.
            if value is None and len(matches) == 1 and i + 1 < len(ocr_lines):
                next_line, next_conf = ocr_lines[i + 1]
                if not _find_all_label_matches(next_line):
                    value = _extract_value_for_kind(meta["kind"], next_line)
                    if value is not None:
                        line_conf = min(conf, next_conf)

            ef = resolve_value(canon, value, line_conf, "flat_line_fallback", line.strip())
            if canon not in results or results[canon].value is None:
                results[canon] = ef
            elif ef.value is not None:
                curr_val = str(results[canon].value).strip()
                is_valve_upgrade = (
                    canon in ("AV", "MV", "TV", "PV")
                    and syn.lower() in ("aortic valve", "mitral valve", "tricuspid valve", "pulmonary valve")
                    and any(w in ef.value.lower() for w in ("trileaflet", "bicuspid", "degenerative", "sclerotic", "adequate", "prolapse", "thickened", "calcif", "stenosis", "mva"))
                )
                if is_valve_upgrade or (curr_val.lower() in ("mild", "moderate", "severe", "trace", "trivial", "normal", "none", "nil", "intact") and ef.value.lower() != curr_val.lower()):
                    results[canon] = ef

    return results


# ---------------------------------------------------------------------------
# Narrative / free-text findings
# ---------------------------------------------------------------------------
def _line_is_heading(line: str) -> bool:
    candidate = normalized_key(line)
    if not candidate:
        return False
    # Clinical finding sentences (e.g. "Valves Structurally Normal.") are findings, not section headings
    if looks_qualitative(line) and not re.search(r":\s*$", line.strip()):
        return False
    for kw in _HEADING_KEYWORDS:
        if candidate == kw or candidate.startswith(kw + ":"):
            return True
        if candidate.startswith(kw + " ") and len(candidate.split()) <= 3:
            return True
        if len(candidate.split()) <= 2 and _fuzzy_ratio(kw, candidate) >= LABEL_MATCH_THRESHOLD:
            return True
    return False


# Headings that introduce the report's own concluding prose. Narrower than _HEADING_KEYWORDS on
# purpose: "findings" and "valves" head measurement blocks on several templates, and this is
# meant to capture only what the cardiologist wrote as their conclusion.
# ---------------------------------------------------------------------------
# Impression / Conclusion -- verbatim, or nothing
# ---------------------------------------------------------------------------
# ONLY these two words count as the heading. "Summary", "Interpretation", "Final Diagnosis",
# "Comment" and "Advice" were deliberately REMOVED: on real templates they head sections that
# are not the cardiologist's conclusion, and treating them as one produced a card that looked
# authoritative while showing something else. If the report has no explicit Impression or
# Conclusion heading, this returns None and the card is hidden -- never a reconstruction
# stitched together from the findings, and never a placeholder.
_IMPRESSION_HEADING_WORDS = frozenset({
    "impression", "impressions", "conclusion", "conclusions",
})

# A heading is a SHORT line. This stops a sentence that merely uses the word ("Findings are
# consistent with the clinical impression of ...") from being read as a section start.
_MAX_HEADING_WORDS = 6

# Glyphs OCR routinely confuses. Used ONLY to compare a candidate token against the two target
# words above -- both strings must be the same length and differ only within one of these
# groups, so this can recognise "IMPRESSlON" without being able to match an unrelated word.
_CONFUSABLE_GLYPHS = ("il1|", "o0", "s5", "b6", "g9", "z2")


def _ocr_equal(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if ca == cb:
            continue
        if not any(ca in group and cb in group for group in _CONFUSABLE_GLYPHS):
            return False
    return True


def _is_impression_heading(line: str) -> bool:
    """True when `line` is an explicit "Impression" / "Conclusion" heading.

    Accepts the real-world forms -- "IMPRESSION", "Impression:", "CONCLUSION :", "Impressions",
    and a qualified heading such as "FINAL DIAGNOSIS (Impression)-" where the word is literally
    present -- while rejecting prose that happens to contain the word.
    """
    probe = normalized_key(line)
    if not probe:
        return False
    words = re.findall(r"[a-z]+", probe)
    if not words or len(probe.split()) > _MAX_HEADING_WORDS:
        return False
    return any(_ocr_equal(w, target)
               for w in words for target in _IMPRESSION_HEADING_WORDS)


def _starts_a_new_section(line: str) -> bool:
    """True for a line that opens a new labelled block, e.g.

        ECHOCARDIOGRAPHY REPORT SUMMARY: CASE 1 - Concordant
        ADVICE : FOLLOW UP ECHO AFTER 1 YEAR

    The test is a SHORT, UPPERCASE label followed by a colon. Body text inside a conclusion is
    sentence-case prose and rarely carries a colon at all, so this ends the section without
    truncating it -- one appendix PDF holds several cases, and without this the first case's
    Conclusion swallowed the entire next report.
    """
    head = line.split(":", 1)[0] if ":" in line else ""
    letters = [c for c in head if c.isalpha()]
    if len(letters) < 3 or len(head.split()) > _MAX_HEADING_WORDS:
        return False
    return all(c.isupper() for c in letters)


def extract_impression_text(raw_text: str) -> Optional[str]:
    """Return the report's own Impression / Conclusion section VERBATIM, or None.

    CONTRACT, and the whole point of this function:
      * Only text that sits under an explicit "Impression" or "Conclusion" heading is returned.
      * It is returned exactly as the OCR read it -- nothing reworded, reordered, summarized,
        de-duplicated or re-wrapped. OCR typos included: this is shown to a clinician as the
        report's own words, and "tidying" it would misrepresent the source.
      * A report with no such heading returns None. Nothing is ever inferred or assembled from
        the findings/valve/chamber sections to stand in for a conclusion the report never made.

    The section runs from its heading to the next recognised section heading, or to the end of
    the document. Reads only `raw_text` -- Stage 1's unfiltered line-for-line output -- rather
    than the reconstructed narrative blocks, which have already been regrouped by row/column
    clustering and are therefore no longer verbatim.
    """
    if not (raw_text or "").strip():
        return None

    lines = raw_text.splitlines()
    start = next((i for i, line in enumerate(lines) if _is_impression_heading(line)), None)
    if start is None:
        return None

    # Stop at the next section heading so a following block (or a page footer that happens to
    # start one) is not presented as part of the conclusion.
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _is_impression_heading(lines[j]):
            continue
        if _line_is_heading(lines[j]) or _starts_a_new_section(lines[j]):
            end = j
            break

    # The heading line is kept: on many templates it carries the finding itself
    # ("IMPRESSION: Normal echocardiogram"), so dropping it would lose real content.
    body = lines[start:end]
    while body and not body[-1].strip():
        body.pop()
    text = "\n".join(body).strip()
    return text or None


def _locate_narrative_sections(full_text: str) -> List[str]:
    """Fuzzy heading match on Conclusion/Impression/Findings/Valve(s)/Summary/Interpretation
    (case-insensitive, tolerant of OCR noise via the same fuzzy-ratio matcher used for
    parameter labels). Everything between a detected heading and the next heading (or end of
    document) is treated as that section's body."""
    lines = full_text.splitlines()
    sections: List[str] = []
    current: List[str] = []
    in_section = False

    heading_re = re.compile(
        r"(?i)^\s*(?:" + "|".join(_HEADING_KEYWORDS) + r")[a-z]*\s*[:\-]?\s*"
    )

    for line in lines:
        if _line_is_heading(line):
            if in_section and current:
                sections.append("\n".join(current))
            current = []
            in_section = True
            remainder = heading_re.sub("", line, count=1).strip()
            if remainder:
                current.append(remainder)
            continue
        if in_section:
            current.append(line)

    if in_section and current:
        sections.append("\n".join(current))
    return sections


_NEGATION_RE = re.compile(r"\b(no|none|nil|absent|without|free of|negative for)\b", re.IGNORECASE)


def _first_term(low: str, terms: Sequence[str]) -> Optional[str]:
    """First vocabulary term present as a whole word. Terms are pre-sorted longest-first so
    "mild to moderate" wins over a bare "mild"."""
    for term in sorted(terms, key=len, reverse=True):
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            return term
    return None


def finding_phrase(clause: str, canon: Optional[str] = None) -> Optional[str]:
    """Normalize a prose clause into the short descriptor a report table would have printed.

      "Mild mitral regurgitation noted."      -> "Mild Regurgitation"
      "The left atrium is moderately dilated" -> "Moderately Dilated"
      "No pericardial effusion."              -> "No Effusion"
      "Normal LV systolic function."          -> "Normal"

    Storing this rather than the whole sentence is what makes a narrative-sourced value
    comparable with a table-sourced one -- the same field ends up holding the same vocabulary
    regardless of which part of the report it came from. The full sentence is preserved in
    meta["source_snippet"] so nothing is lost.
    """
    low = (clause or "").lower()
    if not low.strip():
        return None

    # Canon-specific lesion terms prevent cross-contamination in compound sentences
    # (e.g. "No clots/vegetations/effusion" must not assign "No Effusion" to Clots_Thrombus).
    canon_lesions = None
    if canon == "Clots_Thrombus":
        canon_lesions = ("thrombus", "clot", "clots", "thrombi")
    elif canon == "Pericardial_Effusion":
        canon_lesions = ("effusion", "fluid", "tamponade")
    elif canon == "MV":
        canon_lesions = ("regurgitation", "regurgitant", "stenosis", "stenotic", "sclerosis",
                         "prolapse", "insufficiency", "incompetence", "calcification", "thickening",
                         "mr", "ms", "mvp")
    elif canon == "AV":
        canon_lesions = ("regurgitation", "regurgitant", "stenosis", "stenotic", "sclerosis",
                         "prolapse", "insufficiency", "incompetence", "calcification", "thickening",
                         "bicuspid", "ar", "as", "ai")
    elif canon == "TV":
        canon_lesions = ("regurgitation", "regurgitant", "stenosis", "stenotic", "sclerosis",
                         "prolapse", "insufficiency", "incompetence", "calcification", "thickening",
                         "tr", "ts", "ti")
    elif canon == "PV":
        canon_lesions = ("regurgitation", "regurgitant", "stenosis", "stenotic", "sclerosis",
                         "prolapse", "insufficiency", "incompetence", "calcification", "thickening",
                         "pr", "ps", "pi")

    severity = _first_term(low, SEVERITY_TERMS)
    lesion = _first_term(low, canon_lesions if canon_lesions is not None else LESION_TERMS)
    descriptor = _first_term(low, DESCRIPTOR_TERMS)

    # Negation must be handled explicitly: "no mitral regurgitation" is the OPPOSITE finding to
    # "mitral regurgitation", and silently dropping the "no" would invert the clinical meaning.
    if lesion and _NEGATION_RE.search(low):
        if lesion in ("mr", "tr", "ar", "pr", "ms", "as", "ts", "ps", "ai", "ti", "pi"):
            return f"No {lesion.upper()}"
        return f"No {lesion.title()}"

    if severity and lesion:
        if lesion in ("mr", "tr", "ar", "pr", "ms", "as", "ts", "ps", "mvp", "ai", "ti", "pi"):
            return f"{severity.title()} {lesion.upper()}"
        return f"{severity.title()} {lesion.title()}"
    if lesion:
        if lesion in ("mr", "tr", "ar", "pr", "ms", "as", "ts", "ps", "mvp", "ai", "ti", "pi"):
            return lesion.upper()
        return lesion.title()
    if descriptor:
        # "moderately dilated" -- keep the adverb form the report actually used when present.
        adverb = re.search(r"\b(\w+ly)\s+" + re.escape(descriptor) + r"\b", low)
        if adverb:
            return f"{adverb.group(1).title()} {descriptor.title()}"
        if severity:
            return f"{severity.title()} {descriptor.title()}"
        return descriptor.title()
    return None


def _narrative_field(canon: str, value: str, snippet: str,
                     value_type: Optional[str] = None) -> ExtractedField:
    """Build a narrative-sourced field through the same resolve_value() gate every other path
    uses, then replace the flag reason with one that names prose as the source."""
    ef = resolve_value(canon, value, _NARRATIVE_CONFIDENCE, "narrative", snippet, value_type)
    if ef.value is not None:
        ef.flagged = True
        ef.meta["flagged"] = True
        ef.meta["flag_reason"] = (
            "Extracted from narrative/free-text findings, not a structured table row -- "
            "please confirm against original report."
        )
    return ef


def _groq_narrative_findings(blocks: List[str]) -> Dict[str, ExtractedField]:
    """Ask the text-only model to read findings out of prose. Best-effort: any failure returns
    {} and the deterministic clause matcher below covers the document instead."""
    if not blocks:
        return {}
    try:
        from app.ocr.groq_extractor import extract_narrative_findings
        findings = extract_narrative_findings(blocks, list(PARAMETERS.keys()))
    except Exception:  # noqa: BLE001 -- Groq is an enhancement here, never a dependency
        return {}

    results: Dict[str, ExtractedField] = {}
    for item in findings:
        canon = item.get("key")
        if canon not in PARAMETERS or canon in results:
            continue
        snippet = item.get("evidence") or item.get("value") or ""
        ef = _narrative_field(canon, item.get("value"), snippet, item.get("value_type"))
        if ef.value is not None:
            results[canon] = ef
    return results


def _clause_narrative_findings(full_text: str) -> Dict[str, ExtractedField]:
    """Offline fallback: split the narrative sections into clauses and keep each clause that
    names BOTH a known parameter and a finding/severity term.

    Generalized beyond valves -- any parameter mentioned alongside a descriptor is captured, so
    "the right atrium is enlarged" fills RA_Size just as "mild mitral regurgitation" fills MV.
    """
    results: Dict[str, ExtractedField] = {}
    if not full_text:
        return results

    for section_text in _locate_narrative_sections(full_text):
        for raw_clause in re.split(r"[.;\n]+", section_text):
            clause = raw_clause.strip()
            if not clause:
                continue
            for canon, _syn, _start, _end in _find_all_label_matches(clause):
                if canon in results:
                    continue
                # Do NOT populate strictly numeric fields from qualitative narrative phrases
                if PARAMETERS[canon]["kind"] == "numeric":
                    continue
                phrase = finding_phrase(clause, canon=canon)
                if phrase is None:
                    continue
                ef = _narrative_field(canon, phrase, clause, VALUE_TYPE_QUALITATIVE)
                if ef.value is not None:
                    results[canon] = ef
    return results


_LVOT_NARRATIVE_PATTERNS = [
    # 1. LVOT/LVOTO with maximum/peak/resting/systolic gradient/PSG/PG of [X] mmHg
    re.compile(r'(?i)\b(?:lvoto?|left ventricular outflow tract)\b[^\n.;]*?\b(?:maximum|max|peak|resting|systolic)?\s*(?:gradients?|grads?|psg|pg)\s*(?:of|is|:|–|-|=|@\s*rest|\sat\s*rest)?\s*(\d+(?:\.\d+)?)\s*(?:mm\s*hg|mmhg)?\b'),
    # 2. Maximum/peak/resting/systolic gradient/PSG/PG across/of/in LVOT of [X] mmHg
    re.compile(r'(?i)\b(?:maximum|max|peak|resting|peak systolic|systolic)?\s*(?:gradients?|grads?|psg|pg)\s*(?:of|across|at|in|through)?\s*(?:the\s+)?(?:lvoto?|left ventricular outflow tract)\s*(?:of|is|:|–|-|=|@\s*rest|\sat\s*rest)?\s*(\d+(?:\.\d+)?)\s*(?:mm\s*hg|mmhg)?\b'),
    # 3. LVOT/LVOTO [X] mmHg
    re.compile(r'(?i)\b(?:lvoto?|left ventricular outflow tract)\b[^\n.;]*?\b(\d+(?:\.\d+)?)\s*(?:mm\s*hg|mmhg)\b'),
    # 4. Maximum/peak gradient of [X] mmHg in conclusion/other measurements context
    re.compile(r'(?i)\b(?:maximum|max|peak)\s*gradients?[^\n.;]*?\b(\d+(?:\.\d+)?)\s*(?:mm\s*hg|mmhg)\b'),
]


def _extract_narrative_lvot_gradient(full_text: str) -> Optional[ExtractedField]:
    """Scan narrative text and other measurement sections for LVOT obstruction peak gradients."""
    if not full_text:
        return None
    for line in full_text.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        if re.match(r"^\s*peak\s+gradient\s*\(mmhg\)", line_clean, re.IGNORECASE):
            continue
        for pat in _LVOT_NARRATIVE_PATTERNS:
            m = pat.search(line_clean)
            if m:
                try:
                    val = float(m.group(1))
                    if 5.0 <= val <= 250.0:
                        return resolve_value("LVOT_Peak_Gradient", f"{val:g} mmHg", 95.0, "narrative", line_clean, VALUE_TYPE_NUMERIC)
                except ValueError:
                    continue
    return None


def extract_findings_from_narrative(
    full_text: str, narrative_blocks: Optional[List[str]] = None
) -> Dict[str, ExtractedField]:
    """Qualitative findings from Conclusion/Impression/Findings/Valve(s)/Summary/Interpretation
    prose -- which is where most severity/descriptor findings in a real report actually live.

    Groq reads the full prose block when available (understands context and sentence structure
    better than a regex split). The deterministic clause matcher runs as the fallback.
    """
    if not full_text and not narrative_blocks:
        return {}

    groq_results = _groq_narrative_findings(narrative_blocks or [full_text])
    clause_results = _clause_narrative_findings(full_text)

    # Groq outranks the regex clause matcher, but the clause matcher fills whatever Groq missed.
    combined = dict(clause_results)
    combined.update(groq_results)

    # Extract narrative LVOT peak gradient if present
    lvot_grad_ef = _extract_narrative_lvot_gradient(full_text)
    if lvot_grad_ef and ("LVOT_Peak_Gradient" not in combined or combined["LVOT_Peak_Gradient"].value is None):
        combined["LVOT_Peak_Gradient"] = lvot_grad_ef

    return combined


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def _prose_loses_to_printed_measurement(
    canon: str, narrative_ef: ExtractedField, flat_ef: Optional[ExtractedField]
) -> bool:
    """True when a prose restatement must NOT displace a number or explicit label row.

    Reports habitually say the same thing twice: the measurement block prints "LVEF (est) 40%"
    and the impression says "Mildly reduced left ventricular ejection fraction". Narrative
    outranks the flat-line path in general, but storing "Mildly Reduced" in EF discards the one
    number the report stated.
    Likewise, an explicit line finding like "Clots : Nil" or "Pericardium : Normal" must not be
    displaced by a vague composite sentence in the conclusion.
    """
    if flat_ef is None or flat_ef.value is None:
        return False
    if narrative_ef.value_type != VALUE_TYPE_QUALITATIVE:
        return False
    if PARAMETERS[canon]["kind"] in ("numeric", "chamber_size") and flat_ef.value_type == VALUE_TYPE_NUMERIC:
        return True
    # If flat-line matched an explicit labeled row (e.g. "Clots: Nil", "Pericardium: Normal") with high confidence,
    # it outranks narrative findings from an unrelated or composite sentence.
    if flat_ef.confidence >= OCR_CONFIDENCE_FLAG_THRESHOLD and (flat_ef.value or "").strip().rstrip(".").lower() in ("nil", "none", "normal", "intact", "absent"):
        return True
    return False


def _is_weak(ef: Optional[ExtractedField]) -> bool:
    return ef is None or ef.value is None or ef.confidence < OCR_CONFIDENCE_FLAG_THRESHOLD


def _collect_labels(doc_result: dict) -> List[str]:
    """Every label-ish string in the document, for the one batched semantic-resolution call."""
    labels: List[str] = []
    for row in doc_result.get("table_grid") or []:
        for cell in row:
            text = (cell.get("text") or "").strip()
            if text and not _has_number(text):
                labels.append(text)
    for table in doc_result.get("tables") or []:
        for row in table.get("rows", []):
            for cell in row.get("cells", []):
                text = (cell.get("text") or "").strip()
                if text and not _has_number(text):
                    labels.append(text)
    for ff in doc_result.get("form_fields") or []:
        text = (ff.get("key_text") or "").strip()
        if text:
            labels.append(text)
    return labels


def extract_parameters_structured(doc_result: dict) -> Tuple[Dict[str, ExtractedField], str]:
    """
    Match every parameter against a unified doc_result dict in strict priority order:
      1. structured: table_grid (Stage 1 geometry) and tables, then form_fields on top --
         "table always wins" per spec.
      2. narrative -- fills a canon only when its structured result is missing or below
         OCR_CONFIDENCE_FLAG_THRESHOLD.
      3. flat-line fuzzy matching -- the true last resort (tagged "flat_line_fallback"), tried
         only for whatever neither structured nor narrative could fill. Deliberately lowest
         priority: it's a blunt substring scan that can pick up an incidental label-like word
         inside unrelated prose (e.g. "mitral" inside a narrative sentence), so a more
         deliberate narrative-clause match must be allowed to win over it.
      4. if nothing better was found but a weak/flagged structured result exists, keep it
         anyway (something beats nothing).

    Only canonical dictionary keys ever leave this function -- every path resolves through
    _resolve_label(), which returns a key from PARAMETERS or nothing at all.
    """
    lines = doc_result.get("lines") or []
    tables = doc_result.get("tables") or []
    table_grid = doc_result.get("table_grid") or []
    form_fields = doc_result.get("form_fields") or []
    narrative_blocks = doc_result.get("narrative_blocks") or []
    full_text = doc_result.get("full_text") or ""

    # One Groq call per document for the labels the dictionary could not resolve. Empty (and
    # entirely skipped) when everything already matched, which is the common case.
    semantic_map = build_semantic_label_map(_collect_labels(doc_result))

    # Structured sources in descending reliability. Each only fills what the one above it could
    # not: table_grid carries real cell geometry, so it is never displaced by a flatter source.
    structured: Dict[str, ExtractedField] = match_table_grid(table_grid, semantic_map)
    doppler_matrix = extract_doppler_matrix(doc_result)
    for lower_priority in (doppler_matrix,
                           match_tables(tables, semantic_map),
                           match_form_fields(form_fields, semantic_map)):
        for canon, ef in lower_priority.items():
            if _is_weak(structured.get(canon)):
                structured[canon] = ef

    narrative = extract_findings_from_narrative(full_text, narrative_blocks)
    flat = match_flat_lines(lines)

    results: Dict[str, ExtractedField] = {}
    for canon in set(structured) | set(narrative) | set(flat):
        s = structured.get(canon)
        f = flat.get(canon)
        n = narrative.get(canon)

        # Dedicated valve morphology/comments finding (AV, MV, TV, PV) outranks Doppler table entry
        if canon in ("AV", "MV", "TV", "PV") and f is not None and f.value is not None:
            f_val = str(f.value).strip()
            s_val = str(s.value).strip() if (s is not None and s.value is not None) else None
            is_dedicated_valve_line = bool(re.search(r"\b(?:aortic|mitral|tricuspid|pulmonary)\s+valve\b", f.source_line, re.IGNORECASE))
            has_morphology = any(w in f_val.lower() for w in (
                "trileaflet", "bicuspid", "degenerative", "sclerotic", "sclerosis", "calcif",
                "thickened", "prolapse", "adequate", "stenosis", "mva", "ava"
            ))

            # Only outrank if it's from a dedicated valve section line AND (s is from doppler_table or has morphology)
            if is_dedicated_valve_line and (has_morphology or s is None or s.value is None or s.source == "doppler_table"):
                ef_res = f
                if s_val and s_val.lower() != f_val.lower():
                    ef_res.meta["table_finding"] = s_val
                    ef_res.meta["comments_finding"] = f_val
                    s_sev = N.severity_of(s_val) or ("None" if N.is_absent(s_val) or N.is_explicitly_negated(s_val, "ar", "mr", "tr", "pr") else "")
                    f_sev = N.severity_of(f_val) or ("None" if N.is_absent(f_val) or N.is_explicitly_negated(f_val, "ar", "mr", "tr", "pr") else "")
                    if s_sev and f_sev and s_sev.lower() != f_sev.lower():
                        ef_res.meta["discrepancy_noted"] = True
                        ef_res.meta["discrepancy_detail"] = f"Comments state '{f_val}' while Doppler table states '{s_val}'"
                        ef_res.meta["source_snippet"] = f"{f.source_line} | Doppler Table: {s_val}"
                results[canon] = ef_res
                continue

        if not _is_weak(s):
            results[canon] = s
            continue
        if n is not None and not _prose_loses_to_printed_measurement(canon, n, flat.get(canon)):
            results[canon] = n
            continue
        # A structured value that exists but scored low still beats the flat-line fallback.
        # Flat-line matching is explicitly the least reliable path (it has no cell structure --
        # it extracted 0 of 13 parameters from a real dense 3-column report), so letting it
        # displace a real form_field/table read would trade a good value for a worse guess.
        # Narrative above keeps its documented right to fill low-confidence fields.
        if s is not None and s.value is not None:
            results[canon] = s
            continue
        f = flat.get(canon)
        # Only let the flat-line fallback displace a structured result if it actually found a
        # VALUE. When both are empty, the structured result is the one to keep: it knows WHY it
        # is empty ("source region unreadable/likely blank"), whereas the flat scan only knows
        # it saw a label with nothing after it.
        if f is not None and f.value is not None:
            results[canon] = f
            continue
        if s is not None:
            results[canon] = s
            continue
        if f is not None:
            results[canon] = f

    # Prefer Stage 1's unfiltered raw_text -- that is the verification preview, and it must show
    # every line the OCR engine saw, including the ones filtering discarded.
    raw_text = doc_result.get("raw_text") or full_text or "\n".join(text for text, _ in lines)
    return results, raw_text


def merge_multi_page_results(
    page_results: List[Dict[str, ExtractedField]]
) -> Dict[str, ExtractedField]:
    """Merge extracted fields across multiple pages, keeping the first found."""
    merged: Dict[str, ExtractedField] = {}
    for page in page_results:
        for canon, ef in page.items():
            if canon not in merged:
                merged[canon] = ef
    return merged
