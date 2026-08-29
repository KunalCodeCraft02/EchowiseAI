"""
AGE-GROUP VALUE TABLE — the single lookup behind every blank-parameter dropdown.

Source of truth: `Cardiac_Master_Rule_Engine_v6_Full_Lifespan.md`, §8 (Blank-Parameter Dropdown
Resolution), §8.2 (age-group value tables) and §8.3 (units). Any disagreement between this module
and that document is resolved IN FAVOUR OF THE DOCUMENT — the tables below are a transcription of
it, not an independent clinical source.

WHAT THIS IS FOR
----------------
A field the OCR could not find is blank on the review page. Instead of asking the doctor to type a
number they may not have, the page offers that ONE parameter's own vocabulary (EF gets
Normal/Mild/Moderate/Severe, IVC gets Normal/Dilated-Plethoric, RWMA gets Absent/Present) and
resolves the chosen word into the value band published for THIS PATIENT'S AGE GROUP.

    value_table[field][age_group][option] -> value_or_range

FOUR AGE GROUPS (§8.1)
----------------------
    Children  0-14   pending pediatric review -- NO option resolves to a number (see below)
    Youth     15-24  identical to Adults, by design; there is no separate Youth table
    Adults    25-64  ADULT_TABLE
    Elderly   65+    ADULT_TABLE with the ELDERLY_OVERRIDES applied

CHILDREN ARE DELIBERATELY UNFILLED. Pediatric norms are BSA/z-score dependent, so a fixed mm or
mmHg number would be clinically wrong rather than merely imprecise. For 0-14 every option — even
"Normal" — routes to free-text Custom entry, and the report carries
`pending_pediatric_review: true` with the "pediatric reference ranges not yet clinically
validated" flag. This is the same G1 principle the rest of the engine runs on: an unknown value
is never imputed.

THREE KINDS OF ENTRY
--------------------
    numeric      a published band ("Normal 6-11 mm")           -> resolves to a stored value
    descriptive  a band the guidelines state only in words     -> forces Custom entry
                 ("LV mass mildly above the sex-specific limit")
    qualitative  a word IS the value ("Hypokinetic", "Present") -> stored verbatim

WHAT GETS STORED IN THE PARAMETER COLUMN
----------------------------------------
`"{value} {unit} ({Option})"` — e.g. `34 mm (Normal)`, `29.75 % (Severe)`.

The representative number comes FIRST on purpose: normalize_v4 reads the first number in a field
and the unit token that follows it, so a stored band renders as a real measurement to the rule
engine while the option word stays visible to the clinician and to the text-matching rules
(`is_normal`, `keyword`, `contains`). Storing the raw range instead ("<35 %") would parse as 35 and
grade a severe EF as moderate — the exact class of silent error this module exists to avoid.

The representative number is the midpoint of a closed band, and 15% outside the boundary of an
open one (Normal EF >=55 -> 63.25; Severe EF <35 -> 29.75). It is a stand-in for a measurement the
report never printed, and every consumer is told so: the field is tagged `doctor_dropdown` in
extraction_meta, which caps the confidence score of any finding that rests on it (§9.1).
"""
from typing import Any, Dict, List, Optional, Tuple

import re

VALUE_TABLE_VERSION = "1.0.0"

# ===========================================================================================
# §8.1 Age groups
# ===========================================================================================
CHILDREN, YOUTH, ADULTS, ELDERLY = "children", "youth", "adults", "elderly"

AGE_GROUPS: Dict[str, Dict[str, Any]] = {
    CHILDREN: {
        "label": "Children (0-14 years)",
        "min_age": 0, "max_age": 14,
        # The whole point of this flag: the pediatric table is intentionally EMPTY.
        "pending_pediatric_review": True,
        "force_custom_entry": True,
        # Headline and explanation are separate: the review page and the prediction card both
        # print the headline themselves, and repeating it inside the explanation read as a stutter.
        "flag": "Pediatric reference ranges not yet clinically validated",
        "notice": ("Pediatric norms are BSA/z-score dependent rather than fixed numbers, so no "
                   "dropdown option resolves to a value for this age group — enter the exact "
                   "measured value for every parameter."),
    },
    YOUTH: {
        "label": "Youth (15-24 years)",
        "min_age": 15, "max_age": 24,
        "pending_pediatric_review": False,
        "force_custom_entry": False,
        # Not a copy: the resolver reads the adult table directly for this group.
        "uses_table_of": ADULTS,
        "flag": "", "notice": "",
    },
    ADULTS: {
        "label": "Adults (25-64 years)",
        "min_age": 25, "max_age": 64,
        "pending_pediatric_review": False,
        "force_custom_entry": False,
        "flag": "", "notice": "",
    },
    ELDERLY: {
        "label": "Elderly (65+ years)",
        "min_age": 65, "max_age": 120,
        "pending_pediatric_review": False,
        "force_custom_entry": False,
        "flag": "", "notice": "",
    },
}

# MRE-000 / §5.8: an unextractable age falls back to Adults logic and sets age_unknown.
DEFAULT_AGE_GROUP = ADULTS

def parse_age_years(raw_age: Any) -> Tuple[Optional[float], bool]:
    """Parse raw age string into age in years.
    Handles:
      '6 months', '6 mos', '6 mo', '6m', '06 months' -> 0.5
      '8 weeks', '8 wks', '8w' -> 8/52
      '10 days', '10d' -> 10/365
      '0.5 Y', '0.5 yrs', '0.5 yr', '0.5' -> 0.5
      '12 Y', '12 yrs', '12' -> 12.0
    Returns (age_in_years, is_unknown).
    """
    if raw_age is None:
        return None, True
    text = str(raw_age).strip().lower()
    if not text:
        return None, True

    # Months check (e.g. '06 months', '6 mos', '6 mo', '6m', '6 m')
    m_mo = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|mo\b|m\b)", text)
    if m_mo:
        val = float(m_mo.group(1))
        return round(val / 12.0, 3), False

    # Weeks check
    m_wk = re.search(r"(\d+(?:\.\d+)?)\s*(?:weeks?|wks?|wk\b|w\b)", text)
    if m_wk:
        val = float(m_wk.group(1))
        return round(val / 52.0, 3), False

    # Days check
    m_day = re.search(r"(\d+(?:\.\d+)?)\s*(?:days?|d\b)", text)
    if m_day:
        val = float(m_day.group(1))
        return round(val / 365.0, 3), False

    # Decimal or integer years check (e.g. '0.5 Y', '12 yrs', '45')
    m_yr = re.search(r"(\d+(?:\.\d+)?)\s*(?:y(?:ears?|rs?)?)?", text)
    if m_yr:
        val = float(m_yr.group(1))
        if 0.0 < val <= 120.0:
            return val, False

    return None, True


def resolve_age_group(raw_age: Any) -> Tuple[Optional[str], bool]:
    """'8 years' -> ('children', False); '6 months' -> ('children', False); None -> ('adults', True).

    Returns (age_group, age_unknown). A missing or unusable age never guesses a band: it takes the
    documented Adults fallback and says so, exactly as MRE-000 requires.
    """
    age_years, is_unknown = parse_age_years(raw_age)
    if is_unknown or age_years is None:
        return DEFAULT_AGE_GROUP, True
    if age_years < 15.0:
        return CHILDREN, False
    if age_years <= 24.0:
        return YOUTH, False
    if age_years <= 64.0:
        return ADULTS, False
    return ELDERLY, False


def resolve_sex(raw_gender: Any) -> str:
    """'M' / 'male' -> 'male'. Anything else -> 'unknown'.

    'unknown' is a real answer, not an error: MRE-008 requires the conservative (lower) cutoff and
    an explicit note when sex is not recorded, which is what `_sex_entry` below does.
    """
    text = (str(raw_gender or "")).strip().lower()
    if text.startswith("m"):
        return "male"
    if text.startswith("f"):
        return "female"
    return "unknown"


# ===========================================================================================
# §8.3 Units
# ===========================================================================================
# The unit table itself, documented once and served to the review page so the note under a blank
# field ("...enter an exact value in mmHg") can never disagree with the label beside it.
UNIT_TABLE = [
    {"unit": "%",     "used_for": "Percentages",
     "examples": "Ejection fraction, fractional shortening"},
    {"unit": "mmHg",  "used_for": "Pressure gradients",
     "examples": "AV/MV/TV/LVOT peak gradient, PASP"},
    {"unit": "m/s",   "used_for": "Doppler peak velocities",
     "examples": "AV, MV, TV, LVOT peak velocity"},
    {"unit": "mm",    "used_for": "Linear / wall / chamber dimensions",
     "examples": "IVSd, PWd, valve annulus, pericardial effusion size"},
    {"unit": "cm",    "used_for": "Larger-scale linear dimensions",
     "examples": "IVC diameter, LVOT VTI"},
    {"unit": "cm2",   "used_for": "Valve area",
     "examples": "Mitral valve area, aortic valve area"},
    {"unit": "cm/s",  "used_for": "Tissue / inflow Doppler velocities",
     "examples": "e', a', mitral inflow velocities"},
    {"unit": "ms",    "used_for": "Time intervals",
     "examples": "Deceleration time, IVRT"},
]

# Per-field unit. None means the parameter is a word, not a measurement, and no unit is shown.
FIELD_UNITS: Dict[str, Optional[str]] = {
    "ef": "%",
    "lvidd": "mm", "lvids": "mm", "lvidd_indexed_value": "mm/m2",
    "ivsd": "mm", "ivss": "mm", "pwd": "mm", "pws": "mm",
    "relative_wall_thickness": None,          # unitless ratio (2*PWd/LVIDd)
    "lv_mass": "g/m2",
    "wall_motion": None, "rwma": None,
    "septal_wall_motion": None, "anterior_wall_motion": None, "inferior_wall_motion": None,
    "lateral_wall_motion": None, "posterior_wall_motion": None, "apical_wall_motion": None,
    "la_diameter": "mm", "la_diameter_indexed_value": "cm/m2",
    "ra_size": "mm", "rv_size": "mm", "ivc": "cm",
    "ao_diameter": "mm", "ao_root": "mm", "ao_annulus": "mm", "ao_stj": "mm",
    "av_finding": None, "av_peak_velocity": "m/s", "av_peak_gradient": "mmHg",
    "av_area": "cm2",
    "mv_finding": None, "mv_peak_velocity": "m/s", "mv_peak_gradient": "mmHg",
    "mv_area": "cm2", "mv_annulus": "mm",
    "tv_finding": None, "tv_peak_velocity": "m/s", "tv_peak_gradient": "mmHg",
    "tv_annulus": "mm",
    "pv_finding": None, "pv_peak_velocity": "m/s", "pv_peak_gradient": "mmHg",
    "pv_annulus": "mm",
    "e_a_ratio": None,                         # unitless ratio
    "pasp": "mmHg",
    "lvot_peak_velocity": "m/s", "lvot_peak_gradient": "mmHg", "lvot_vti": "cm",
    # Effusion is offered as a size grade, not a measurement; §8.3's "mm" applies only when a
    # report prints an actual effusion depth, which is an OCR-extracted value, not a dropdown one.
    "pericardial_effusion": None,
    "clots_thrombus": None,
    "bsa": "m2", "height": "cm", "weight": "kg",
}


# ===========================================================================================
# §8.0 Dropdown options — each parameter's OWN vocabulary
# ===========================================================================================
# Deliberately NOT one shared Normal/Low/Moderate/High list: "Moderate LVIDd" is not a thing a
# cardiologist writes, and offering it would invite a value the guidelines never published.
CUSTOM = "Custom"

DROPDOWN_OPTIONS: Dict[str, List[str]] = {
    "ef": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "lvidd": ["Normal", "Mildly Dilated", "Severely Dilated", CUSTOM],
    "lvids": ["Normal", "Dilated", CUSTOM],
    "lvidd_indexed_value": ["Normal", "Dilated", CUSTOM],
    "ivsd": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "ivss": ["Normal", "Thickened", CUSTOM],
    "pwd": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "pws": ["Normal", "Thickened", CUSTOM],
    "relative_wall_thickness": ["Normal", "Increased", CUSTOM],
    "lv_mass": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "rwma": ["Absent", "Present"],
    "septal_wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "anterior_wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "inferior_wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "lateral_wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "posterior_wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "apical_wall_motion": ["Normal", "Hypokinetic", "Akinetic", "Dyskinetic"],
    "la_diameter": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "la_diameter_indexed_value": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "ra_size": ["Normal", "Mildly Dilated", "Severely Dilated", CUSTOM],
    "rv_size": ["Normal", "Mildly Dilated", "Severely Dilated", CUSTOM],
    "ivc": ["Normal", "Dilated/Plethoric", CUSTOM],
    "ao_diameter": ["Normal", "Dilated", "Aneurysmal", CUSTOM],
    "ao_root": ["Normal", "Dilated", CUSTOM],
    "ao_annulus": ["Normal", "Dilated", CUSTOM],
    "ao_stj": ["Normal", "Dilated", CUSTOM],
    "av_finding": ["Normal", "Mild Regurgitation", "Moderate Regurgitation",
                   "Severe Regurgitation", "Mild Stenosis", "Moderate Stenosis",
                   "Severe Stenosis", "Mixed Disease", "Calcified/Sclerotic", CUSTOM],
    "av_peak_velocity": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "av_peak_gradient": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "av_area": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "mv_finding": ["Normal", "Mild Regurgitation", "Moderate Regurgitation",
                   "Severe Regurgitation", "Mild Stenosis", "Moderate Stenosis",
                   "Severe Stenosis", "Prolapse", "Mixed Disease", "Calcified/Sclerotic", CUSTOM],
    "mv_peak_velocity": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "mv_peak_gradient": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "mv_area": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "mv_annulus": ["Normal", "Dilated", CUSTOM],
    "tv_finding": ["Normal", "Mild Regurgitation", "Moderate Regurgitation",
                   "Severe Regurgitation", "Stenosis", CUSTOM],
    "tv_peak_velocity": ["Normal", "Elevated", CUSTOM],
    "tv_peak_gradient": ["Normal", "Elevated", CUSTOM],
    "tv_annulus": ["Normal", "Dilated", CUSTOM],
    "pv_finding": ["Normal", "Mild Regurgitation", "Moderate Regurgitation",
                   "Severe Regurgitation", "Stenosis", CUSTOM],
    "pv_peak_velocity": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "pv_peak_gradient": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "pv_annulus": ["Normal", "Dilated", CUSTOM],
    "e_a_ratio": ["Low (Impaired Relaxation)", "Normal", "Pseudonormal",
                  "High (Restrictive)", CUSTOM],
    "pasp": ["Normal", "Mild", "Moderate", "Severe", CUSTOM],
    "lvot_peak_velocity": ["Normal", "Moderate", "Severe", CUSTOM],
    "lvot_peak_gradient": ["Normal", "Moderate", "Severe", CUSTOM],
    "lvot_vti": ["Normal", "Reduced", CUSTOM],
    "pericardial_effusion": ["None/Trace", "Small", "Moderate", "Large"],
    "clots_thrombus": ["No Clots", "Clots Present"],
    "bsa": [CUSTOM],
    "height": [CUSTOM],
    "weight": [CUSTOM],
}


# ===========================================================================================
# Entry constructors
# ===========================================================================================
def _n(low: Optional[float], high: Optional[float], display: str,
       note: Optional[str] = None) -> Dict[str, Any]:
    """A published numeric band. `low`/`high` are inclusive where the display text says so."""
    return {"kind": "numeric", "low": low, "high": high, "display": display, "note": note}


def _d(display: str, note: Optional[str] = None) -> Dict[str, Any]:
    """A band the guidelines describe in words only -- no number can be assigned.

    Selecting one of these forces Custom entry rather than inventing a cutoff. §6.5.4 of the master
    document is explicit that thresholds unsupported by the guideline hierarchy are not to be
    fabricated, and that applies here as much as to the rules themselves.
    """
    return {"kind": "descriptive", "low": None, "high": None, "display": display, "note": note}


def _q(stored: str, display: Optional[str] = None) -> Dict[str, Any]:
    """A word that IS the value. Stored verbatim so the text-matching rules read it unchanged."""
    return {"kind": "qualitative", "low": None, "high": None,
            "display": display or stored, "stored": stored, "note": None}


def _sex(male: Dict[str, Any], female: Dict[str, Any]) -> Dict[str, Any]:
    """A band with published sex-specific limits (MRE-000 / MRE-008)."""
    return {"kind": "by_sex", "male": male, "female": female}


# --- shared qualitative option sets --------------------------------------------------------
_WALL_MOTION = {
    "Normal": _q("Normal"),
    "Hypokinetic": _q("Hypokinetic"),
    "Akinetic": _q("Akinetic"),
    "Dyskinetic": _q("Dyskinetic"),
}

# The stored text carries the lesion word the valve rules match on ("regurgitation", "stenosis",
# "prolapse", "calcified") AND the severity word severity_of() grades on, so a dropdown selection
# produces exactly the finding a report would have printed.
def _valve_options(*, prolapse: bool = False, graded_stenosis: bool = True) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "Normal": _q("Normal"),
        "Mild Regurgitation": _q("Mild regurgitation"),
        "Moderate Regurgitation": _q("Moderate regurgitation"),
        "Severe Regurgitation": _q("Severe regurgitation"),
    }
    if graded_stenosis:
        opts.update({
            "Mild Stenosis": _q("Mild stenosis"),
            "Moderate Stenosis": _q("Moderate stenosis"),
            "Severe Stenosis": _q("Severe stenosis"),
            "Mixed Disease": _q("Mixed disease -- regurgitation and stenosis"),
            "Calcified/Sclerotic": _q("Calcified / sclerotic"),
        })
    else:
        opts["Stenosis"] = _q("Stenosis")
    if prolapse:
        opts["Prolapse"] = _q("Prolapse")
    return opts


# ===========================================================================================
# §8.2a ADULT / YOUTH TABLE
# ===========================================================================================
# Groups A and B are numeric; D/E/F are qualitative and identical in every age group.
ADULT_TABLE: Dict[str, Dict[str, Any]] = {
    # --- Group A: Normal / Mild / Moderate / Severe -----------------------------------------
    "ef": {
        "Normal": _n(55, None, ">= 55 %"),
        "Mild": _n(45, 54, "45-54 %"),
        "Moderate": _n(35, 44, "35-44 %"),
        "Severe": _n(None, 35, "< 35 %"),
    },
    "ivsd": {
        "Normal": _n(6, 11, "6-11 mm"),
        "Mild": _n(12, 14, "12-14 mm"),
        # The source table prints "--" here: mild runs straight into severe at 15 mm.
        "Moderate": _d("no separate moderate band is published for septal thickness",
                       "Mild runs to 14 mm and severe begins at 15 mm -- enter the exact value."),
        "Severe": _n(15, None, ">= 15 mm"),
    },
    "pwd": {
        "Normal": _n(6, 11, "6-11 mm"),
        "Mild": _n(12, 14, "12-14 mm"),
        "Moderate": _d("no separate moderate band is published for posterior wall thickness",
                       "Mild runs to 14 mm and severe begins at 15 mm -- enter the exact value."),
        "Severe": _n(15, None, ">= 15 mm"),
    },
    "lv_mass": {
        "Normal": _sex(_n(None, 115, "<= 115 g/m2 (male)"),
                       _n(None, 95, "<= 95 g/m2 (female)")),
        "Mild": _d("mildly above the sex-specific normal limit (95 F / 115 M g/m2)"),
        "Moderate": _d("no separate moderate band is published for LV mass index"),
        "Severe": _d("markedly above the sex-specific normal limit (95 F / 115 M g/m2)"),
    },
    "la_diameter": {
        "Normal": _n(None, 40, "<= 40 mm"),
        "Mild": _n(41, 46, "41-46 mm"),
        "Moderate": _n(47, 52, "47-52 mm"),
        "Severe": _n(52, None, "> 52 mm"),
    },
    "la_diameter_indexed_value": {
        "Normal": _n(None, 2.3, "<= 2.3 cm/m2"),
        "Mild": _n(2.4, 2.6, "2.4-2.6 cm/m2"),
        "Moderate": _n(2.7, 2.9, "2.7-2.9 cm/m2"),
        "Severe": _n(3.0, None, ">= 3.0 cm/m2"),
    },
    "la_volume_index": {
        "Normal": _n(None, 34, "<= 34 mL/m2"),
        "Mild": _n(35, 41, "35-41 mL/m2"),
        "Moderate": _n(42, 48, "42-48 mL/m2"),
        "Severe": _n(49, None, ">= 49 mL/m2"),
    },
    "av_peak_velocity": {
        "Normal": _n(None, 2.0, "< 2.0 m/s"),
        "Sclerosis": _n(2.0, 2.5, "2.0-2.5 m/s (sclerosis, not stenosis)"),
        "Mild": _n(2.6, 2.9, "2.6-2.9 m/s"),
        "Moderate": _n(3.0, 3.9, "3.0-3.9 m/s"),
        "Severe": _n(4.0, None, ">= 4.0 m/s"),
    },
    "av_peak_gradient": {
        "Normal": _n(None, 10, "< 10 mmHg"),
        "Mild": _n(10, 19, "10-19 mmHg"),
        "Moderate": _n(20, 39, "20-39 mmHg"),
        "Severe": _n(40, None, ">= 40 mmHg"),
    },
    "av_area": {
        "Normal": _n(3.0, 4.0, "3.0-4.0 cm2"),
        "Mild": _n(1.5, None, "> 1.5 cm2"),
        "Moderate": _n(1.0, 1.5, "1.0-1.5 cm2"),
        "Severe": _n(None, 1.0, "< 1.0 cm2"),
    },
    "mv_peak_velocity": {
        "Normal": _n(None, 1.2, "< 1.2 m/s"),
        "Mild": _n(1.2, 1.5, "1.2-1.5 m/s"),
        "Moderate": _n(1.5, 2.5, "1.5-2.5 m/s"),
        "Severe": _n(2.5, None, "> 2.5 m/s"),
    },
    "mv_peak_gradient": {
        "Normal": _n(None, 2, "< 2 mmHg"),
        "Mild": _n(2, 5, "2-5 mmHg"),
        "Moderate": _n(6, 10, "6-10 mmHg"),
        "Severe": _n(10, None, "> 10 mmHg"),
    },
    "mv_area": {
        "Normal": _n(4.0, 6.0, "4-6 cm2"),
        "Mild": _n(1.5, 2.5, "1.5-2.5 cm2"),
        "Moderate": _n(1.0, 1.5, "1.0-1.5 cm2"),
        "Severe": _n(None, 1.0, "<= 1.0 cm2"),
    },
    "pv_peak_velocity": {
        "Normal": _n(None, 1.0, "< 1.0 m/s"),
        "Mild": _n(1.0, 3.0, "1.0-3.0 m/s"),
        "Moderate": _n(3.0, 4.0, "3.0-4.0 m/s"),
        "Severe": _n(4.0, None, "> 4.0 m/s"),
    },
    "pv_peak_gradient": {
        "Normal": _n(None, 10, "< 10 mmHg"),
        "Mild": _n(10, 20, "10-20 mmHg"),
        "Moderate": _n(21, 40, "21-40 mmHg"),
        "Severe": _n(40, None, "> 40 mmHg"),
    },
    "pasp": {
        "Normal": _n(None, 35, "<= 35 mmHg"),
        "Mild": _n(36, 45, "36-45 mmHg"),
        "Moderate": _n(46, 60, "46-60 mmHg"),
        "Severe": _n(60, None, "> 60 mmHg"),
    },
    "lvot_peak_velocity": {
        "Normal": _n(None, 2.5, "< 2.5 m/s"),
        "Moderate": _n(2.5, 3.5, "2.5-3.5 m/s"),
        "Severe": _n(3.5, None, "> 3.5 m/s"),
    },


    
    "lvot_peak_gradient": {
        "Normal": _n(None, 30, "< 30 mmHg"),
        "Moderate": _n(30, 50, "30-50 mmHg"),
        "Severe": _n(50, None, "> 50 mmHg"),
    },

    # --- Group B: Normal / option 2 / option 3 ----------------------------------------------
    "lvidd": {
        "Normal": _sex(_n(42, 58, "42-58 mm (male)"), _n(38, 52, "38-52 mm (female)")),
        "Mildly Dilated": _sex(_n(59, 63, "59-63 mm (male)"), _n(53, 56, "53-56 mm (female)")),
        "Severely Dilated": _sex(_n(63, None, "> 63 mm (male)"), _n(56, None, "> 56 mm (female)")),
    },
    "lvids": {
        "Normal": _n(25, 40, "25-40 mm"),
        "Dilated": _n(40, None, "> 40 mm"),
    },
    "lvidd_indexed_value": {
        "Normal": _n(22, 31, "22-31 mm/m2"),
        "Dilated": _n(31, None, "> 31 mm/m2"),
    },
    "ivss": {
        "Normal": _n(8, 16, "8-16 mm"),
        "Thickened": _n(16, None, "> 16 mm"),
    },
    "pws": {
        "Normal": _n(8, 16, "8-16 mm"),
        "Thickened": _n(16, None, "> 16 mm"),
    },
    "relative_wall_thickness": {
        "Normal": _n(0.32, 0.42, "0.32-0.42"),
        "Increased": _n(0.42, None, "> 0.42"),
    },
    "ra_size": {
        "Normal": _n(None, 44, "<= 44 mm"),
        "Mildly Dilated": _n(45, 49, "45-49 mm"),
        "Severely Dilated": _n(50, None, ">= 50 mm"),
    },
    "rv_size": {
        "Normal": _n(None, 42, "<= 42 mm"),
        "Mildly Dilated": _n(43, 47, "43-47 mm"),
        "Severely Dilated": _n(48, None, ">= 48 mm"),
    },
    "ivc": {
        "Normal": _n(None, 2.1, "< 2.1 cm, collapses > 50 %"),
        "Dilated/Plethoric": _n(2.1, None, ">= 2.1 cm, collapses < 50 %"),
    },
    "ao_diameter": {
        "Normal": _n(None, 40, "<= 40 mm"),
        "Dilated": _n(41, 45, "41-45 mm"),
        "Aneurysmal": _n(45, None, "> 45 mm"),
    },
    "ao_root": {
        "Normal": _sex(_n(30, 37, "30-37 mm (male)"), _n(27, 33, "27-33 mm (female)")),
        "Dilated": _sex(_n(37, None, "> 37 mm (male)"), _n(33, None, "> 33 mm (female)")),
    },
    "ao_annulus": {
        "Normal": _sex(_n(20, 29, "20-29 mm (male)"), _n(18, 26, "18-26 mm (female)")),
        "Dilated": _sex(_n(29, None, "> 29 mm (male)"), _n(26, None, "> 26 mm (female)")),
    },
    "ao_stj": {
        "Normal": _sex(_n(23, 32, "23-32 mm (male)"), _n(20, 29, "20-29 mm (female)")),
        "Dilated": _sex(_n(32, None, "> 32 mm (male)"), _n(29, None, "> 29 mm (female)")),
    },
    "mv_annulus": {
        "Normal": _n(26, 35, "26-35 mm"),
        "Dilated": _n(35, None, "> 35 mm"),
    },
    "tv_annulus": {
        "Normal": _n(28, 34, "28-34 mm"),
        "Dilated": _n(34, None, "> 34 mm"),
    },
    "pv_annulus": {
        "Normal": _n(18, 24, "18-24 mm"),
        "Dilated": _n(24, None, "> 24 mm"),
    },
    "tv_peak_velocity": {
        "Normal": _n(None, 2.8, "< 2.8 m/s"),
        "Elevated": _n(2.8, None, ">= 2.8 m/s"),
    },
    "tv_peak_gradient": {
        "Normal": _n(None, 31, "< 31 mmHg"),
        "Elevated": _n(31, None, ">= 31 mmHg"),
    },
    "lvot_vti": {
        "Normal": _n(18, 22, "18-22 cm"),
        "Reduced": _n(None, 18, "< 18 cm"),
    },
    "lvot_diameter": {
        "Normal": _n(1.8, 2.2, "1.8-2.2 cm"),
    },
    "tapse": {
        "Normal": _n(17, None, ">= 17 mm"),
        "Reduced": _n(None, 17, "< 17 mm"),
    },

    # --- Group C: diastolic filling pattern ---------------------------------------------------
    "e_a_ratio": {
        "Low (Impaired Relaxation)": _n(None, 0.8, "< 0.8"),
        "Normal": _n(0.8, 2.0, "0.8-2.0"),
        # Same numeric window as Normal -- the LA is what separates them, which is why the stored
        # value carries the qualifier and MRE-018's corroboration requirement still applies.
        "Pseudonormal": _n(0.8, 2.0, "0.8-2.0 with LA dilation",
                           "Pseudonormal filling is Normal-looking by E/A alone. MRE-018 requires "
                           "corroboration (LA size, e', TRV) before it can be graded."),
        "High (Restrictive)": _n(2.0, None, "> 2.0"),
    },
    # DELIBERATELY NOT TABLED: e_e_ratio and deceleration_time.
    # Neither has a standalone cut-off to offer. Under the ASE/EACVI 2016 diastolic function
    # algorithm both are inputs to an INTEGRATED assessment -- read alongside E/A ratio, LA volume
    # index and TR velocity -- and their meaning inverts with the E/A pattern they accompany: a
    # short deceleration time is restrictive filling with a high E/A but normal with a low one.
    # A dropdown resolving either to a single number would state a grade the guideline does not
    # support on that measurement alone, which §6.5.4 forbids. Enter these as exact values.

    # --- Groups D/E/F: qualitative, age-independent -------------------------------------------
    "wall_motion": dict(_WALL_MOTION),
    "septal_wall_motion": dict(_WALL_MOTION),
    "anterior_wall_motion": dict(_WALL_MOTION),
    "inferior_wall_motion": dict(_WALL_MOTION),
    "lateral_wall_motion": dict(_WALL_MOTION),
    "posterior_wall_motion": dict(_WALL_MOTION),
    "apical_wall_motion": dict(_WALL_MOTION),
    "rwma": {
        "Absent": _q("Absent -- no regional wall motion abnormality"),
        "Present": _q("Present"),
    },
    "av_finding": _valve_options(),
    "mv_finding": _valve_options(prolapse=True),
    "tv_finding": _valve_options(graded_stenosis=False),
    "pv_finding": _valve_options(graded_stenosis=False),
    "pericardial_effusion": {
        # The stored wording is what the pericardial rule and the exercise screen grade on, so
        # each option carries both its size word and the severity word it corresponds to.
        "None/Trace": _q("None / trace -- no significant pericardial effusion"),
        "Small": _q("Small pericardial effusion (mild)"),
        "Moderate": _q("Moderate pericardial effusion"),
        "Large": _q("Large pericardial effusion"),
    },
    "clots_thrombus": {
        "No Clots": _q("No clots / thrombus seen"),
        "Clots Present": _q("Clots / thrombus present"),
    },
}


# ===========================================================================================
# §8.2b ELDERLY OVERRIDES
# ===========================================================================================
# Only the parameters whose published bands genuinely differ at 65+. Everything absent from this
# dict is identical to the adult table -- consistent with §6.5's rule that age must not silently
# move a threshold that has no age-specific evidence behind it.
ELDERLY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "ivsd": {
        "Normal": _n(6, 12, "6-12 mm"),
        "Mild": _n(13, 14, "13-14 mm"),
        "Moderate": _d("no separate moderate band is published for septal thickness",
                       "Mild runs to 14 mm and severe begins at 15 mm -- enter the exact value."),
        "Severe": _n(15, None, ">= 15 mm"),
    },
    "pwd": {
        "Normal": _n(6, 12, "6-12 mm"),
        "Mild": _n(13, 14, "13-14 mm"),
        "Moderate": _d("no separate moderate band is published for posterior wall thickness",
                       "Mild runs to 14 mm and severe begins at 15 mm -- enter the exact value."),
        "Severe": _n(15, None, ">= 15 mm"),
    },
    "la_diameter": {
        "Normal": _n(None, 42, "<= 42 mm"),
        "Mild": _n(43, 47, "43-47 mm"),
        "Moderate": _n(48, 52, "48-52 mm"),
        "Severe": _n(52, None, "> 52 mm"),
    },
    "la_diameter_indexed_value": {
        "Normal": _n(None, 2.4, "<= 2.4 cm/m2"),
        "Mild": _n(2.5, 2.7, "2.5-2.7 cm/m2"),
        "Moderate": _n(2.8, 3.0, "2.8-3.0 cm/m2"),
        "Severe": _n(3.1, None, ">= 3.1 cm/m2"),
    },
    "pasp": {
        "Normal": _n(None, 40, "<= 40 mmHg"),
        "Mild": _n(41, 50, "41-50 mmHg"),
        "Moderate": _n(51, 65, "51-65 mmHg"),
        "Severe": _n(65, None, "> 65 mmHg"),
    },
    "lvidd": {
        "Normal": _sex(_n(40, 56, "40-56 mm (male)"), _n(36, 50, "36-50 mm (female)")),
        "Mildly Dilated": _sex(_n(57, 61, "57-61 mm (male)"), _n(51, 54, "51-54 mm (female)")),
        "Severely Dilated": _sex(_n(61, None, "> 61 mm (male)"), _n(54, None, "> 54 mm (female)")),
    },
    "lvids": {
        "Normal": _n(24, 39, "24-39 mm"),
        "Dilated": _n(39, None, "> 39 mm"),
    },
    "lvidd_indexed_value": {
        "Normal": _n(21, 30, "21-30 mm/m2"),
        "Dilated": _n(30, None, "> 30 mm/m2"),
    },
    "ivss": {
        "Normal": _n(8, 17, "8-17 mm"),
        "Thickened": _n(17, None, "> 17 mm"),
    },
    "pws": {
        "Normal": _n(8, 17, "8-17 mm"),
        "Thickened": _n(17, None, "> 17 mm"),
    },
    "relative_wall_thickness": {
        "Normal": _n(0.32, 0.45, "0.32-0.45"),
        "Increased": _n(0.45, None, "> 0.45"),
    },
    "ao_diameter": {
        "Normal": _n(None, 42, "<= 42 mm"),
        "Dilated": _n(43, 47, "43-47 mm"),
        "Aneurysmal": _n(47, None, "> 47 mm"),
    },
    "ao_root": {
        "Normal": _sex(_n(31, 39, "31-39 mm (male)"), _n(28, 35, "28-35 mm (female)")),
        "Dilated": _sex(_n(39, None, "> 39 mm (male)"), _n(35, None, "> 35 mm (female)")),
    },
    "ao_stj": {
        "Normal": _sex(_n(24, 34, "24-34 mm (male)"), _n(21, 31, "21-31 mm (female)")),
        "Dilated": _sex(_n(34, None, "> 34 mm (male)"), _n(31, None, "> 31 mm (female)")),
    },
    "mv_annulus": {
        "Normal": _n(26, 36, "26-36 mm"),
        "Dilated": _n(36, None, "> 36 mm"),
    },
    "tv_annulus": {
        "Normal": _n(28, 35, "28-35 mm"),
        "Dilated": _n(35, None, "> 35 mm"),
    },
    "tv_peak_velocity": {
        "Normal": _n(None, 2.9, "< 2.9 m/s"),
        "Elevated": _n(2.9, None, ">= 2.9 m/s"),
    },
    "tv_peak_gradient": {
        "Normal": _n(None, 34, "< 34 mmHg"),
        "Elevated": _n(34, None, ">= 34 mmHg"),
    },
    "lvot_vti": {
        "Normal": _n(16, 21, "16-21 cm"),
        "Reduced": _n(None, 16, "< 16 cm"),
    },
    "e_a_ratio": {
        "Low (Impaired Relaxation)": _n(None, 0.6, "< 0.6"),
        "Normal": _n(0.6, 1.0, "0.6-1.0"),
        "Pseudonormal": _n(0.6, 1.0, "0.6-1.0 with LA dilation",
                           "Pseudonormal filling is Normal-looking by E/A alone. MRE-018 requires "
                           "corroboration (LA size, e', TRV) before it can be graded."),
        "High (Restrictive)": _n(2.0, None, "> 2.0"),
    },
}


# ===========================================================================================
# Athlete-adjusted reference bands (Outdoor Sports Person)
# ===========================================================================================
# NOT part of VALUE_TABLE / the blank-field dropdown -- these never change a dropdown option, a
# unit or the standard bands DISEASE rules and the §9 scores are computed against (those stay on
# the standard tables everywhere, unconditionally). This is a SEPARATE resolution path consulted
# only by _rule_athlete / _rule_exercise_safety in rules_v4.py, so a physiologically enlarged
# athlete heart is not screened out of sport or exercise on the same numbers that (correctly)
# still flag it for HCM/LVH surveillance as a disease finding.
#
# Bands only exist for Adults/Youth (one shared table, as elsewhere in this module) and Elderly --
# Children are never athlete-adjusted (constraint: no changes to 0-14 logic).
#
# ivsd/pwd deliberately use the LOWER (endurance-athlete) ceiling rather than the higher
# strength-athlete one published in the source spec (15/16 mm M, 12/13 mm F): this system collects
# no endurance-vs-strength distinction, and the lower ceiling is the conservative choice -- a
# strength athlete just past it lands in the mandatory gray-zone check below instead of being
# silently waved through.
ATHLETE_ADJUSTABLE_FIELDS = frozenset({
    "ivsd", "pwd", "lvidd", "lv_mass", "ra_size", "rv_size", "e_a_ratio",
})

# Parameters the spec is explicit must NEVER resolve against an athlete table, regardless of the
# flag -- documented here for the badge/audit trail; the resolver below already only ever adjusts
# ATHLETE_ADJUSTABLE_FIELDS, so this list is not separately enforced, only reported.
ATHLETE_ALWAYS_STANDARD_FIELDS = frozenset({
    "av_finding", "av_peak_velocity", "av_peak_gradient", "av_area",
    "mv_finding", "mv_peak_velocity", "mv_peak_gradient", "mv_area", "mv_annulus",
    "tv_finding", "tv_peak_velocity", "tv_peak_gradient", "tv_annulus",
    "pv_finding", "pv_peak_velocity", "pv_peak_gradient", "pv_annulus",
    "pasp", "lvot_peak_velocity", "lvot_peak_gradient", "lvot_vti",
    "pericardial_effusion", "clots_thrombus",
    "ao_diameter", "ao_root", "ao_annulus", "ao_stj", "ivc",
    "wall_motion", "septal_wall_motion", "anterior_wall_motion", "inferior_wall_motion",
    "lateral_wall_motion", "posterior_wall_motion", "apical_wall_motion", "rwma",
    "la_diameter", "la_diameter_indexed_value", "lvids", "lvidd_indexed_value",
    "ivss", "pws", "ef",
})

# {sex: {"<field>_high": ceiling, ...}} -- only the bound the athlete table actually moves.
# Whatever bound is absent here falls back to the standard table's own bound for that side, so an
# ivsd BELOW 6 mm is still thin even for an athlete: only the upper ceiling is relaxed.
_ATHLETE_ADULT_BANDS: Dict[str, Dict[str, float]] = {
    "male": {"ivsd_high": 15.0, "pwd_high": 15.0, "lvidd_high": 66.0, "lv_mass_high": 150.0,
             "ra_size_high": 49.0, "rv_size_high": 47.0, "e_a_ratio_low": 0.6},
    "female": {"ivsd_high": 12.0, "pwd_high": 12.0, "lvidd_high": 61.0, "lv_mass_high": 125.0,
               "ra_size_high": 46.0, "rv_size_high": 44.0, "e_a_ratio_low": 0.6},
}
# Elderly-athlete: midpoint between adult-athlete and standard elderly, per spec. ra_size/rv_size
# and e_a_ratio have no published elderly-athlete numbers in the source spec beyond "smaller
# allowance than younger athletes" / no explicit figure -- interpolated conservatively (smaller
# than the adult-athlete allowance) rather than left unadjusted.
_ATHLETE_ELDERLY_BANDS: Dict[str, Dict[str, float]] = {
    "male": {"ivsd_high": 14.0, "pwd_high": 14.0, "lvidd_high": 60.0, "lv_mass_high": 135.0,
             "ra_size_high": 47.0, "rv_size_high": 45.0, "e_a_ratio_low": 0.5},
    "female": {"ivsd_high": 12.0, "pwd_high": 12.0, "lvidd_high": 55.0, "lv_mass_high": 110.0,
               "ra_size_high": 45.0, "rv_size_high": 43.0, "e_a_ratio_low": 0.5},
}


def _athlete_band_table(age_group: Optional[str]) -> Optional[Dict[str, Dict[str, float]]]:
    if age_group in (ADULTS, YOUTH):
        return _ATHLETE_ADULT_BANDS
    if age_group == ELDERLY:
        return _ATHLETE_ELDERLY_BANDS
    return None                                    # Children: never athlete-adjusted


def athlete_adjusted_band(field: str, age_group: Optional[str],
                          sex: Any) -> Optional[Dict[str, Optional[float]]]:
    """The athlete-adjusted {low, high} Normal band for one field, or None when it doesn't apply.

    None means: not one of the eight adjustable fields, age group has no athlete table (Children,
    or an unrecognised value), or the field genuinely has no athlete-adjusted bound. Sex-unknown
    takes the conservative (female) band, the same MRE-008 convention `_sex_entry` applies to the
    standard tables.
    """
    if field not in ATHLETE_ADJUSTABLE_FIELDS:
        return None
    table = _athlete_band_table(age_group)
    if table is None:
        return None
    sex_key = sex if sex in ("male", "female") else "female"
    band = table.get(sex_key)
    if not band:
        return None
    standard = normal_band(field, age_group, sex) or {}
    low = band.get(f"{field}_low", standard.get("low"))
    high = band.get(f"{field}_high", standard.get("high"))
    if low is None and high is None:
        return None
    return {"low": low, "high": high}


def athlete_eccentric_lvidd(age_group: Optional[str], sex: Any,
                            lvidd_value: Optional[float]) -> bool:
    """True when an LVIDd measurement sits at/inside the athlete-adjusted band -- i.e. chamber
    enlargement proportional to an athletic remodeling pattern rather than pathological dilation.

    Used to gate the RWT eccentric-pattern exception: an elevated relative wall thickness reads as
    normal for an athlete ONLY when the ventricle is enlarged to match, never on RWT alone.
    """
    if lvidd_value is None:
        return False
    band = athlete_adjusted_band("lvidd", age_group, sex)
    if not band or band.get("high") is None:
        return False
    low = band.get("low") if band.get("low") is not None else 0.0
    return low <= lvidd_value <= band["high"]


def athlete_gray_zone(field: str, age_group: Optional[str], sex: Any,
                      value: Optional[float]) -> bool:
    """True when `value` sits strictly between the STANDARD ceiling and the athlete-adjusted
    ceiling for this age+gender band -- the zone the mandatory gray-zone check screens (§ Athlete's
    Heart vs HCM). Only meaningful for ivsd/pwd. Runs unconditionally, independent of the
    patient's own athlete flag: this is a differential-diagnosis screen, not a clearance.
    """
    if value is None or field not in ("ivsd", "pwd"):
        return False
    standard = normal_band(field, age_group, sex) or {}
    athlete = athlete_adjusted_band(field, age_group, sex) or {}
    std_high, ath_high = standard.get("high"), athlete.get("high")
    if std_high is None or ath_high is None or ath_high <= std_high:
        return False
    return std_high < value < ath_high


def _elderly_table() -> Dict[str, Dict[str, Any]]:
    table = {field: dict(options) for field, options in ADULT_TABLE.items()}
    for field, options in ELDERLY_OVERRIDES.items():
        table[field] = dict(options)
    return table


# value_table[field][age_group][option] -> entry. Children is empty BY DESIGN, not by omission.
VALUE_TABLE: Dict[str, Dict[str, Dict[str, Any]]] = {}
for _field in DROPDOWN_OPTIONS:
    VALUE_TABLE[_field] = {
        CHILDREN: {},
        YOUTH: ADULT_TABLE.get(_field, {}),
        ADULTS: ADULT_TABLE.get(_field, {}),
        ELDERLY: _elderly_table().get(_field, {}),
    }


# ===========================================================================================
# Resolution
# ===========================================================================================
def _round(value: float) -> float:
    """Two decimals, and no trailing .0 on a whole number."""
    rounded = round(value + 1e-9, 2)
    return int(rounded) if float(rounded).is_integer() else rounded


def representative_value(low: Optional[float], high: Optional[float]) -> Optional[float]:
    """The single number that stands in for a band.

    Closed band -> midpoint. Open band -> 15% outside the stated boundary, so the value lands
    unambiguously inside the band rather than exactly on the cut-off, where a `<`/`<=` difference
    would flip the grade.
    """
    if low is not None and high is not None:
        return _round((low + high) / 2.0)
    if high is not None:
        return _round(high * 0.85)
    if low is not None:
        return _round(low * 1.15)
    return None


def _sex_entry(entry: Dict[str, Any], sex: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Pick the sex-specific band, or the conservative one plus a note when sex is unrecorded."""
    if entry.get("kind") != "by_sex":
        return entry, None
    if sex in ("male", "female"):
        return entry[sex], None
    # MRE-008: with no recorded sex, take the lower (female) limits and say so.
    return entry["female"], ("Patient sex is not recorded — the conservative (female) limits were "
                            "applied. Record sex, or enter the exact value, for a sex-specific band.")


def resolve(field: str, option: str, age_group: Optional[str] = None,
            sex: Any = None) -> Dict[str, Any]:
    """Resolve one dropdown selection into the value the parameter column should hold.

    Always returns a dict. `requires_custom=True` means the caller must collect a free-text value
    instead of storing anything: the pediatric band, an option the guidelines only describe in
    words, and the literal "Custom" option all land there.
    """
    group = age_group or DEFAULT_AGE_GROUP
    sex_key = resolve_sex(sex)
    unit = FIELD_UNITS.get(field)
    meta = AGE_GROUPS.get(group, AGE_GROUPS[DEFAULT_AGE_GROUP])

    out: Dict[str, Any] = {
        "field": field, "option": option, "age_group": group, "sex": sex_key, "unit": unit,
        "kind": None, "display": None, "value": None, "low": None, "high": None,
        "stored_value": None, "requires_custom": False, "note": None,
        "pending_pediatric_review": bool(meta.get("pending_pediatric_review")),
    }

    if option == CUSTOM:
        out["kind"] = "custom"
        out["requires_custom"] = True
        out["note"] = custom_note(field)
        return out

    # Children: the table is empty on purpose. Even "Normal" resolves to nothing.
    if meta.get("force_custom_entry"):
        out["kind"] = "pending_pediatric_review"
        out["requires_custom"] = True
        out["note"] = meta.get("notice")
        return out

    entry = VALUE_TABLE.get(field, {}).get(group, {}).get(option)
    if entry is None:
        out["kind"] = "unmapped"
        out["requires_custom"] = True
        out["note"] = ("No published band for this option in this age group — enter the exact "
                       "value.")
        return out

    entry, sex_note = _sex_entry(entry, sex_key)
    out["kind"] = entry["kind"]
    out["display"] = entry.get("display")
    out["note"] = " ".join(n for n in (entry.get("note"), sex_note) if n) or None

    if entry["kind"] == "qualitative":
        out["stored_value"] = entry["stored"]
        return out

    if entry["kind"] == "descriptive":
        out["requires_custom"] = True
        if not out["note"]:
            out["note"] = ("This grade is defined in words rather than a number — enter the exact "
                           f"value{f' in {unit}' if unit else ''}.")
        return out

    low, high = entry.get("low"), entry.get("high")
    value = representative_value(low, high)
    out.update({"low": low, "high": high, "value": value})
    if value is None:
        out["requires_custom"] = True
        return out
    out["stored_value"] = f"{value} {unit} ({option})" if unit else f"{value} ({option})"
    return out


def custom_note(field: str) -> str:
    """The line shown under a blank field, naming the unit an exact value must be entered in."""
    unit = FIELD_UNITS.get(field)
    options = [o for o in DROPDOWN_OPTIONS.get(field, []) if o != CUSTOM]
    choices = "/".join(options[:4]) if options else "an option"
    if unit:
        return (f"Leave as {choices}, or select Custom to enter an exact value in {unit}.")
    return (f"Leave as {choices}, or select the finding this report describes.")


def normal_band(field: str, age_group: Optional[str] = None,
                sex: Any = None) -> Optional[Dict[str, Any]]:
    """The age-resolved Normal band for one field, or None when it has no numeric one.

    This is what the prediction/severity scores measure distance from (§9.2/§9.3), so a finding in
    a 70-year-old is scored against the 65+ boundary rather than a flat adult one.
    """
    group = age_group or DEFAULT_AGE_GROUP
    entry = VALUE_TABLE.get(field, {}).get(group, {}).get("Normal")
    if entry is None:
        return None
    entry, _ = _sex_entry(entry, resolve_sex(sex))
    if entry.get("kind") != "numeric":
        return None
    return {"low": entry.get("low"), "high": entry.get("high"), "display": entry.get("display")}


def excess_over_normal(field: str, value: Optional[float], age_group: Optional[str] = None,
                       sex: Any = None) -> Optional[float]:
    """How far past the age-resolved Normal band a measurement sits, as a fraction of that bound.

    0.0 means inside the normal band; 0.3 means 30% beyond the boundary it crossed. None when the
    field has no numeric normal band or no value was measured -- the caller then falls back to the
    documented fixed default rather than inventing a distance.
    """
    if value is None:
        return None
    band = normal_band(field, age_group, sex)
    if band is None:
        return None
    low, high = band["low"], band["high"]
    if high is not None and value > high and high != 0:
        return (value - high) / abs(high)
    if low is not None and value < low and low != 0:
        return (low - value) / abs(low)
    return 0.0


# ===========================================================================================
# Frontend payload
# ===========================================================================================
def _slim(resolved: Dict[str, Any]) -> Dict[str, Any]:
    """The four things the page actually renders, without the lookup keys it already knows."""
    out = {"value": resolved["stored_value"], "display": resolved["display"]}
    if resolved["requires_custom"]:
        out["custom"] = True
    if resolved["note"]:
        out["note"] = resolved["note"]
    return out


def dropdown_config() -> Dict[str, Any]:
    """Everything the review page needs to render and resolve a blank-field dropdown.

    Pre-resolved on the server (per age group, per sex where sex matters) so the browser only ever
    performs a lookup. The resolution logic therefore lives in exactly one place, and a table edit
    here cannot leave the page showing a stale band.

    Children are not enumerated: every option in that band resolves the same way (no value, custom
    entry forced), so the page reads `age_groups.children.force_custom_entry` instead of carrying
    a table of identical empty answers.
    """
    table: Dict[str, Any] = {}
    for field, options in DROPDOWN_OPTIONS.items():
        per_group: Dict[str, Any] = {}
        for group in (YOUTH, ADULTS, ELDERLY):
            sex_specific = any(
                VALUE_TABLE[field][group].get(o, {}).get("kind") == "by_sex" for o in options)
            if sex_specific:
                per_group[group] = {"by_sex": {
                    sex: {o: _slim(resolve(field, o, group, sex)) for o in options}
                    for sex in ("male", "female", "unknown")
                }}
            else:
                per_group[group] = {o: _slim(resolve(field, o, group)) for o in options}
        table[field] = per_group

    return {
        "version": VALUE_TABLE_VERSION,
        "age_groups": AGE_GROUPS,
        "default_age_group": DEFAULT_AGE_GROUP,
        "custom_option": CUSTOM,
        "options": DROPDOWN_OPTIONS,
        "units": FIELD_UNITS,
        "unit_table": UNIT_TABLE,
        "custom_notes": {f: custom_note(f) for f in DROPDOWN_OPTIONS},
        "value_table": table,
    }
