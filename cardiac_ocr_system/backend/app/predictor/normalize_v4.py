"""
v4.0 normalization layer for the Cardiac Disease Prediction Rule Engine.

Turns the extraction pipeline's stored strings into the typed, unit-normalized values the v4.0
rules are written against. Nothing here reaches back into the OCR pipeline -- it only reads what
that pipeline already stored.

THE FOUR GENERAL PRINCIPLES (G1-G4)
-----------------------------------
G1  MISSING IS MISSING. A null, "not in report", "not mentioned" or "enter manually" is never
    imputed, defaulted or treated as zero/normal. It returns None and every rule that needs it
    simply does not fire. This is the single most important property of the module: in a
    clinical tool, "we don't know" must never silently become "it's fine".
G2  DUAL EVALUATION. The same field arrives as a number ("45", "1.2") or as a categorical word
    ("Normal", "Enlarged", "Thickened"), because 2D-Echo reports genuinely print either. Every
    accessor therefore answers both questions: number_of() and keyword_of().
G3  CONCLUSION VERIFICATION. conclusion_text is matched case-insensitively as a substring, so a
    diagnosis the cardiologist wrote in prose still fires even when the numbers are absent.
G4  SYNONYM FOLDING. Vocabulary is standardized before any rule runs (see _SYNONYMS).

MAGNITUDE-BASED UNIT INFERENCE
------------------------------
The pipeline stores values with units where the report printed them ("4.31 cm", "43.1 mm"), but
a doctor's manual correction or a bare table cell can arrive unitless. The rules are written in
millimetres, so a unitless number is resolved by magnitude -- 1.0-9.9 is centimetres, 10-99 is
already millimetres, and anything over 100 is biologically impossible for a chamber dimension and
is discarded rather than guessed at.
"""
import re
from typing import Optional, List, Tuple

# --- G1: tokens that mean "no value" ------------------------------------------------------
# "not in report" is the review page's wording for an unfilled parameter; "enter manually" is
# its placeholder. Both must read as absent, not as data.
_MISSING_TOKENS = {
    "", "-", "--", "n/a", "na", "nil", "none", "null", "nan",
    "not detected", "not in report", "not mentioned", "not found", "not seen",
    "not visualized", "not visualised", "not assessed", "enter manually", "unknown",
}

# --- G4: synonym dictionary ---------------------------------------------------------------
# Longest phrases first so "normal lv function" folds before the bare word "normal" can.
_SYNONYMS = [
    ("normal lv function", "normal ef"),
    ("good lv function", "normal ef"),
    ("preserved lv function", "normal ef"),
    ("depressed lv function", "lv dysfunction"),
    ("poor lv function", "lv dysfunction"),
    ("reduced ef", "lv dysfunction"),
    ("lv dysfunction", "lv dysfunction"),
    ("plethoric ivc", "raised right atrial pressure"),
    ("dilated ivc", "raised right atrial pressure"),
    ("collapse <50%", "raised right atrial pressure"),
    ("collapse < 50%", "raised right atrial pressure"),
    ("hypokinesia", "hypokinetic"),
    ("hypokinetic", "hypokinetic"),
    ("akinesia", "akinetic"),
    ("akinetic", "akinetic"),
    ("dyskinesia", "dyskinetic"),
    ("dyskinetic", "dyskinetic"),
    ("dilatation", "dilated"),
    ("dilated", "dilated"),
    ("enlarged", "dilated"),
    ("physiological", "mild"),
    ("trace", "mild"),
]

_NUM_RE = re.compile(r"(-?\d+(?:\.\d+)?)")
_UNIT_RE = re.compile(r"(cm/sec|cm/s|m/sec|m/s|mmhg|mm\s?hg|cm2|cm/m2|g/m2|mm|cm|%)", re.IGNORECASE)


def clean(raw) -> Optional[str]:
    """G1. Normalize whitespace per line (preserving newlines) and return None for missing tokens."""
    if raw is None:
        return None
    lines = [" ".join(line.split()) for line in str(raw).splitlines()]
    text = "\n".join(l for l in lines if l).strip()
    if not text or text.lower() in _MISSING_TOKENS:
        return None
    return text


def keyword(raw) -> Optional[str]:
    """G2 + G4. Lowercased text with the synonym dictionary applied, or None."""
    text = clean(raw)
    if text is None:
        return None
    lowered = text.lower()
    for source, target in _SYNONYMS:
        if source in lowered:
            return target
    return lowered


# --- Clinical Negation Detection -----------------------------------------------------------
# Common negation terms and phrases used in clinical text and echocardiography reports.
# Ordered with multi-word phrases first so composite phrases are recognized cleanly.
NEGATION_TERMS = [
    # Multi-word negation phrases
    "no evidence of",
    "no sign of",
    "no signs of",
    "no significant",
    "no obvious",
    "no definite",
    "no gross",
    "absence of",
    "negative for",
    "ruled out",
    "rule out",
    "free of",
    "not seen",
    "not detected",
    "not found",
    "not visualized",
    "not visualised",
    "not present",
    "not noted",
    "not identified",
    "not apparent",
    # Single-word negation terms
    "no",
    "not",
    "without",
    "absent",
    "denies",
    "nil",
    "none",
    "neither",
    "never",
    "negative",
]

POST_NEGATION_TERMS = [
    "ruled out",
    "absent",
    "none",
    "nil",
    "negative",
    "not seen",
    "not detected",
    "not present",
    "not found",
    "not visualized",
    "not visualised",
]

# Patterns that terminate the scope of a preceding negation
# (sentence terminators, contrasting conjunctions, or comma followed by a positive severity word)
CLAUSE_BREAKERS_RE = re.compile(
    r"(?:[\.;!\?\n\r]+|\b(?:but|however|although|yet|except|except for|despite)\b|,\s*(?=(?:mild|moderate|severe|trace|trivial|significant|grade)\b))",
    re.IGNORECASE,
)


def is_negated_match(text: str, match_start: int, match_end: int, max_word_window: int = 5) -> bool:
    """Checks whether the keyword match at [match_start:match_end] in text is negated.

    1. Preceding check: Look at the text in the same clause immediately before match_start.
       If a negation term appears within max_word_window words before the match, it is negated.
    2. Following check: Look at the text in the same clause immediately after match_end.
       If a post-negation term appears within 3 words after the match, it is negated.
    """
    preceding_all = text[:match_start]

    # Find the last clause break in preceding text
    clause_matches = list(CLAUSE_BREAKERS_RE.finditer(preceding_all))
    if clause_matches:
        last_break = clause_matches[-1].end()
        clause_preceding = preceding_all[last_break:]
    else:
        clause_preceding = preceding_all

    clause_preceding_clean = clause_preceding.strip()
    if clause_preceding_clean:
        lowered_pre = clause_preceding.lower()
        for phrase in NEGATION_TERMS:
            if " " in phrase:
                p_pattern = r"\b" + re.escape(phrase) + r"\b"
                m = re.search(p_pattern, lowered_pre)
                if m:
                    between = lowered_pre[m.end():]
                    words_between = re.findall(r"\b\w+\b", between)
                    if len(words_between) <= max_word_window:
                        return True
            else:
                words = re.findall(r"\b\w+\b", lowered_pre)
                window_words = words[-max_word_window:] if len(words) > max_word_window else words
                if phrase in window_words:
                    return True

    # Post-negation check (after match)
    following_all = text[match_end:]
    first_break = CLAUSE_BREAKERS_RE.search(following_all)
    clause_following = following_all[:first_break.start()] if first_break else following_all
    clause_following_clean = clause_following.strip()
    if clause_following_clean:
        lowered_post = clause_following.lower()
        for phrase in POST_NEGATION_TERMS:
            p_pattern = r"\b" + re.escape(phrase) + r"\b"
            m = re.search(p_pattern, lowered_post)
            if m:
                between = lowered_post[:m.start()]
                words_between = re.findall(r"\b\w+\b", between)
                if len(words_between) <= 2:
                    return True

    return False


def find_matches(text: str, needle: str, is_word_boundary: bool = False) -> List[Tuple[int, int]]:
    """Find all (start, end) index tuples of needle in text."""
    lowered = text.lower()
    n_low = needle.lower().strip()
    if not n_low:
        return []
    # Always enforce word boundaries on alpha-numeric needles to prevent word-prefix clashing (e.g. 'severe as' in 'severe asymmetric')
    pattern = r"\b" + re.escape(n_low) + r"\b"
    return [(m.start(), m.end()) for m in re.finditer(pattern, lowered)]


def contains(raw, *needles: str) -> bool:
    """G3. Case-insensitive substring test, returning True only if at least one needle
    has a NON-NEGATED match. Safe on missing values."""
    text = clean(raw)
    if text is None:
        return False
    for needle in needles:
        if not needle:
            continue
        matches = find_matches(text, needle, is_word_boundary=False)
        for start, end in matches:
            if not is_negated_match(text, start, end):
                return True
    return False


def contains_word(raw, *tokens: str) -> bool:
    """Case-insensitive token test requiring word boundaries (\b), returning True only if
    at least one token has a NON-NEGATED match. Safe on missing values."""
    text = clean(raw)
    if text is None:
        return False
    for tok in tokens:
        if not tok:
            continue
        matches = find_matches(text, tok, is_word_boundary=True)
        for start, end in matches:
            if not is_negated_match(text, start, end):
                return True
    return False


def is_explicitly_negated(raw, *needles: str) -> bool:
    """True if ANY match for the given needles in raw is explicitly negated."""
    text = clean(raw)
    if text is None:
        return False
    for needle in needles:
        if not needle:
            continue
        is_word = len(needle.strip()) <= 3
        matches = find_matches(text, needle, is_word_boundary=is_word)
        for start, end in matches:
            if is_negated_match(text, start, end):
                return True
    return False



def _number_and_unit(raw):
    text = clean(raw)
    if text is None:
        return None, None
    match = _NUM_RE.search(text)
    if not match:
        return None, None
    try:
        value = float(match.group(1))
    except ValueError:
        return None, None
    unit_match = _UNIT_RE.search(text[match.end():])
    unit = unit_match.group(1).lower().replace(" ", "").replace("sec", "s") if unit_match else None
    return value, unit


def to_mm(raw) -> Optional[float]:
    """Chamber dimensions and wall thicknesses, in millimetres.

    An explicit unit is always believed. A unitless number is resolved by magnitude:
        1.0 - 9.9   centimetres  -> x10
        10  - 99    already millimetres
        >= 100      biologically impossible for these structures -> discarded, NOT clamped.
    Discarding rather than clamping matters: a clamped value would silently become a plausible
    measurement and could trigger a diagnosis off an OCR error.
    """
    value, unit = _number_and_unit(raw)
    if value is None or value < 0:
        return None
    if unit == "mm":
        return value
    if unit == "cm":
        return value * 10.0
    if unit in ("%", "mmhg", "m/s", "cm/s", "cm2", "g/m2", "cm/m2"):
        return None                      # a different quantity entirely
    if 1.0 <= value <= 9.9:
        return value * 10.0
    if 10.0 <= value <= 99.0:
        return value
    return None


def to_percent(raw) -> Optional[float]:
    """Ejection fraction. A unitless 10-85 is read as a percentage."""
    value, unit = _number_and_unit(raw)
    if value is None:
        return None
    if unit == "%":
        return value
    if unit is not None:
        return None
    return value if 10.0 <= value <= 85.0 else None


def to_velocity_ms(raw) -> Optional[float]:
    """Doppler peak velocity in m/s. Unitless 0.5-6.0 is m/s; a larger unitless number in the
    gradient band is NOT a velocity and is rejected rather than guessed."""
    value, unit = _number_and_unit(raw)
    if value is None:
        return None
    if unit == "m/s":
        return value
    if unit == "cm/s":
        return value / 100.0
    if unit is not None:
        return None
    return value if 0.5 <= value <= 6.0 else None


def to_gradient_mmhg(raw) -> Optional[float]:
    """Doppler peak gradient in mmHg. Unitless 5-120 is a gradient."""
    value, unit = _number_and_unit(raw)
    if value is None:
        return None
    if unit == "mmhg":
        return value
    if unit is not None:
        return None
    return value if 5.0 <= value <= 120.0 else None


def to_ratio(raw) -> Optional[float]:
    """Unitless ratios (E/A, RWT). Rejects anything carrying a unit."""
    value, unit = _number_and_unit(raw)
    if value is None or unit is not None:
        return None
    return value


def to_index(raw) -> Optional[float]:
    """BSA-indexed values (LV mass index g/m2, LA/BSA cm/m2)."""
    value, unit = _number_and_unit(raw)
    if value is None:
        return None
    if unit in (None, "g/m2", "cm/m2"):
        return value
    return None


# --- shared vocabularies used by the rules -------------------------------------------------
NORMAL_WORDS = ("normal", "wnl", "unremarkable", "no abnormality", "structurally normal", "none/trace", "no clots", "no clot")
ABNORMAL_MOTION_WORDS = ("hypokinetic", "akinetic", "dyskinetic", "abnormal")
SEVERITY_ORDER = ("trace", "mild", "mild-moderate", "moderate", "moderate-severe", "severe")


def severity_of(raw) -> Optional[str]:
    """Highest severity word present, in the canonical casing the UI displays.

    Scans worst-first so "moderate-severe" is not mistaken for plain "moderate", and so a
    finding naming two grades reports the worse one -- under-reporting severity is the more
    dangerous error.
    """
    text = clean(raw)
    if text is None:
        return None
    lowered = text.lower()
    for word in reversed(SEVERITY_ORDER):
        if word in lowered:
            return word.title()
    return None


def is_normal(raw) -> bool:
    """True only when the text positively says normal. Missing is NOT normal (G1)."""
    text = clean(raw)
    if text is None:
        return False
    lowered = text.lower()
    if any(w in lowered for w in ("abnormal", "not normal")):
        return False
    return any(w in lowered for w in NORMAL_WORDS)


def is_absent(raw) -> bool:
    """True when a finding is positively reported as absent ("No pericardial effusion", "No Clots", "None/Trace")."""
    text = clean(raw)
    if text is None:
        return False
    lowered = text.lower()
    return any(w in lowered for w in ("no ", "nil", "absent", "none", "not seen", "not detected", "not visualized", "without", "no clots", "none/trace"))
