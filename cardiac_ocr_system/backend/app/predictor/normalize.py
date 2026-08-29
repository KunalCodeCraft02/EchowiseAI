"""
Normalization helpers that turn raw stored parameter strings (as saved by the
OCR/extraction layer, e.g. "58%", "1.3 cm", "13 mm", "Mildly Dilated") into
typed values the rule engine can compare against clinical thresholds.

Every parser returns None on anything unparseable/missing. Callers must never
treat None as zero/false/normal — see the MISSING DATA PROTOCOL note in
app/predictor/rules.py.
"""
import re
from typing import NamedTuple, Optional

_NUM_UNIT_RE = re.compile(r"(-?\d+\.?\d*)\s*(mmhg|mm hg|mm|cm|%)?", re.IGNORECASE)

# Strings that mean "no value", not a value. "not in report" is the wording the review page now
# uses for an empty parameter; "not detected" is kept because older rows and some report
# templates ([XXX] Facility prints it literally) still carry it.
_MISSING_TOKENS = {"", "none", "n/a", "na", "not detected", "not in report", "not found",
                   "unknown", "-", "--", "nil"}


def _clean(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in _MISSING_TOKENS:
        return None
    return s or None


def parse_percent(raw) -> Optional[float]:
    """EF-style values, e.g. '58%', '58'."""
    s = _clean(raw)
    if s is None:
        return None
    m = _NUM_UNIT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# E/A ratios and PASP (mmHg) use the same bare-number extraction as EF.
parse_ratio = parse_percent
parse_pressure_mmhg = parse_percent


def parse_length_mm(raw) -> Optional[float]:
    """
    Length-like measurements (LVIDd/LVIDs/IVSd/PWd/IVSs/PWs/LA/Ao/IVC).
    Rule thresholds in this engine are expressed in millimeters. Stored
    values may carry an explicit unit ('cm'/'mm') or none at all — when the
    unit is missing, infer it from magnitude: every parameter in this family
    has a valid clinical range of roughly 0.5-8 cm (5-80 mm), so a bare
    number <=15 is assumed to be centimeters (matches parameter_dict.py's
    valid_range values) and a bare number >15 is assumed to already be mm.
    """
    s = _clean(raw)
    if s is None:
        return None
    m = _NUM_UNIT_RE.search(s)
    if not m:
        return None
    try:
        number = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    if unit == "mm":
        return number
    if unit == "cm":
        return number * 10
    return number * 10 if number <= 15 else number


_VELOCITY_RE = re.compile(r"(-?\d+\.?\d*)\s*(cm/sec|cm/s|m/sec|m/s)?", re.IGNORECASE)


def parse_velocity_ms(raw) -> Optional[float]:
    """
    Doppler peak velocities, returned in METRES PER SECOND (the unit every guideline
    threshold in this engine is expressed in: AV Vmax >=4.0 m/s = severe aortic stenosis,
    TR Vmax >2.8 m/s = elevated pulmonary pressure).

    The extractor already normalizes these to m/s, but reports print both m/s and cm/s and a
    bare number can still arrive from a manual doctor correction. When the unit is missing,
    infer from magnitude: no physiological Doppler velocity reaches 15 m/s, so a bare number
    above that is cm/s.
    """
    s = _clean(raw)
    if s is None:
        return None
    m = _VELOCITY_RE.search(s)
    if not m:
        return None
    try:
        number = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower().replace("sec", "s")
    if unit == "cm/s":
        return number / 100.0
    if unit == "m/s":
        return number
    return number / 100.0 if number > 15 else number


_AREA_RE = re.compile(r"(-?\d+\.?\d*)\s*(cm2|cm²)?", re.IGNORECASE)


def parse_area_cm2(raw) -> Optional[float]:
    """Valve areas in cm^2 (mitral valve area grades mitral stenosis severity)."""
    s = _clean(raw)
    if s is None:
        return None
    m = _AREA_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


_ENLARGED_WORDS = ("enlarg", "dilat", "increased")
_NORMAL_SIZE_WORDS = ("normal", "wnl", "unremarkable", "not dilated", "not enlarged")


def classify_chamber_size(raw, enlarged_mm_threshold: float) -> Optional[str]:
    """
    Classify a chamber-size reading (RA_Size / RV_Size) into 'Enlarged' or
    'Normal'. Reports vary: some give a qualitative read, some give a raw
    diameter. Qualitative wording wins when present; otherwise falls back to
    comparing the parsed measurement against a documented ASE reference
    upper-limit-of-normal diameter for that chamber, used only when the
    report supplied a bare number with no descriptive wording.
    """
    s = _clean(raw)
    if s is None:
        return None
    low = s.lower()
    if any(w in low for w in _NORMAL_SIZE_WORDS):
        return "Normal"
    if any(w in low for w in _ENLARGED_WORDS):
        return "Enlarged"
    mm = parse_length_mm(s)
    if mm is None:
        return None
    return "Enlarged" if mm > enlarged_mm_threshold else "Normal"


_ABSENT_WORDS = ("absent", "none", "no effusion", "no thrombus", "no clot", "not seen", "negative")
_PRESENT_WORDS = ("present", "trace", "trivial", "mild", "moderate", "severe", "seen", "positive", "small", "large")


def classify_presence(raw) -> Optional[bool]:
    """
    Generic present/absent classifier for qualitative findings
    (Pericardial Effusion, Clots/Thrombus). True = present, False = absent.
    """
    s = _clean(raw)
    if s is None:
        return None
    low = s.lower()
    if any(w in low for w in _ABSENT_WORDS):
        return False
    if any(w in low for w in _PRESENT_WORDS):
        return True
    return None


class ValveFinding(NamedTuple):
    abnormal: Optional[bool]   # None = unknown/unparseable
    kind: Optional[str]        # "regurgitation" | "stenosis" | None
    severity: Optional[str]    # "mild" | "moderate" | "severe" | None


_SEVERITY_WORDS = [
    ("severe", "severe"), ("critical", "severe"),
    ("moderate", "moderate"), ("mod ", "moderate"), ("mod-", "moderate"),
    ("mild", "mild"), ("minimal", "mild"), ("minor", "mild"),
]
_STENOSIS_WORDS = ("stenosis", "sclerosis", "stenotic")
_REGURG_WORDS = ("regurgitation", "regurg", "prolapse", "incompetence", "insufficiency")
_NORMAL_VALVE_WORDS = ("normal", "no significant", "physiologic", "unremarkable")
# "Trivial"/"trace" regurgitation is a normal physiological variant seen in a
# majority of healthy hearts (especially TR/PR) — not a graded disease state,
# unlike "mild" which is a genuine (if lowest) disease grade.
_PHYSIOLOGIC_WORDS = ("trivial", "trace")

# Short-form clinical abbreviations (MR/AR/TR/PR = regurgitation, MS/AS/TS/PS
# = stenosis). Safe to match loosely here because this text is already just
# the short qualitative remainder captured after a specific valve label
# (e.g. "AV:"), not a full sentence.
_REGURG_ABBR_RE = re.compile(r"\b(MR|AR|TR|PR)\b", re.IGNORECASE)
_STENOSIS_ABBR_RE = re.compile(r"\b(MS|AS|TS|PS)\b", re.IGNORECASE)


def classify_valve(raw) -> ValveFinding:
    s = _clean(raw)
    if s is None:
        return ValveFinding(None, None, None)
    low = s.lower()

    kind = None
    if any(w in low for w in _STENOSIS_WORDS) or _STENOSIS_ABBR_RE.search(s):
        kind = "stenosis"
    elif any(w in low for w in _REGURG_WORDS) or _REGURG_ABBR_RE.search(s):
        kind = "regurgitation"

    # Trivial/trace regurgitation is a normal physiological variant, not
    # clinically significant disease — checked before severity grading so it
    # can't be misread as a "mild" disease state.
    if kind == "regurgitation" and any(w in low for w in _PHYSIOLOGIC_WORDS) \
            and not any(w in low for w in ("mild", "moderate", "severe")):
        return ValveFinding(False, None, None)

    severity = None
    for word, sev in _SEVERITY_WORDS:
        if word in low:
            severity = sev
            break

    if kind is None and severity is None:
        if any(w in low for w in _NORMAL_VALVE_WORDS):
            return ValveFinding(False, None, None)
        return ValveFinding(None, None, None)  # unrecognized text -> unknown

    if kind is None:
        # Abnormality/severity wording present but no explicit valve lesion
        # type -> default to regurgitation, the most common echo finding.
        kind = "regurgitation"

    return ValveFinding(True, kind, severity)
