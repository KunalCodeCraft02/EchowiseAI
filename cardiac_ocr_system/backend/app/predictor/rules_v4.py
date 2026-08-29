"""
CARDIAC DISEASE PREDICTION RULE ENGINE (v4.0 ENHANCED)

A Clinical Decision Support System. Every output requires physician review; nothing here is a
diagnosis, and a 2D Echo cannot demonstrate coronary blockage directly -- the ischemia rules are
explicitly proxies that recommend further workup.

SCOPE: this module reads the parameters the extraction pipeline already stored and produces
predictions. It does not touch extraction, preprocessing or the parameter columns themselves.

WHAT MAKES THIS SAFE
--------------------
* MISSING NEVER BECOMES NORMAL (G1). A rule whose inputs are absent does not fire, and is
  reported as `evaluated=False` rather than as a negative result. "Normal Heart" in particular
  requires POSITIVE evidence on every criterion it checks -- an empty report cannot be declared
  a normal heart.
* EVERY PREDICTION CARRIES ITS EVIDENCE. supporting_points names the parameter, its value and
  the threshold crossed, so a clinician can audit the reasoning rather than trust it.
* CONCLUSION TEXT IS FIRST-CLASS (G3). When the cardiologist wrote the diagnosis in prose, that
  is stronger evidence than a derived number, and the supporting point says so explicitly.

The public entry point is evaluate_v4(params) -> dict.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.predictor import normalize_v4 as N
from app.predictor import value_table as V

ENGINE_VERSION = "4.0"

DISCLAIMER = (
    "Clinical Decision Support output only -- not an automated diagnosis. A 2D Echo cannot "
    "directly demonstrate coronary artery blockage. Every finding requires physician review and "
    "correlation with clinical history."
)

# Maps the CardiacReport column names the extraction pipeline writes onto the v4.0 spec's
# parameter names. This mapping is the ONLY coupling to the extraction side, and it is read-only.
PARAM_MAP = {
    "ejection_fraction": "ef",
    "lvidd": "lvidd", "lvids": "lvids", "lvidd_bsa": "lvidd_indexed_value",
    "ivsd": "ivsd", "ivss": "ivss", "pwd": "pwd", "pws": "pws",
    "rwt": "relative_wall_thickness", "lv_mass_index": "lv_mass",
    "wall_motion": "wall_motion", "rwma": "rwma",
    "septal_wall_motion": "septal_wall_motion", "anterior_wall_motion": "anterior_wall_motion",
    "inferior_wall_motion": "inferior_wall_motion", "lateral_wall_motion": "lateral_wall_motion",
    "posterior_wall_motion": "posterior_wall_motion", "apical_wall_motion": "apical_wall_motion",
    "la_diameter": "la_diameter", "la_diameter_bsa": "la_diameter_indexed_value",
    "ra_size": "ra_size", "rv_size": "rv_size", "ivc": "ivc",
    "aortic_diameter": "ao_diameter", "aortic_root": "ao_root",
    "aortic_annulus": "ao_annulus", "stj": "ao_stj",
    "aortic_valve_finding": "av_finding",
    "av_peak_velocity": "av_peak_velocity", "av_peak_gradient": "av_peak_gradient",
    "mitral_valve_finding": "mv_finding",
    "mv_peak_velocity": "mv_peak_velocity", "mv_peak_gradient": "mv_peak_gradient",
    "mitral_valve_area": "mv_area", "mitral_annulus": "mv_annulus",
    "tricuspid_valve_finding": "tv_finding",
    "tv_peak_velocity": "tv_peak_velocity", "tv_peak_gradient": "tv_peak_gradient",
    "tricuspid_annulus": "tv_annulus",
    "pulmonary_valve_finding": "pv_finding",
    "pv_peak_velocity": "pv_peak_velocity", "pv_peak_gradient": "pv_peak_gradient",
    "pulmonic_annulus": "pv_annulus",
    "ea_ratio": "e_a_ratio", "pasp": "pasp",
    "lvot_peak_velocity": "lvot_peak_velocity", "lvot_peak_gradient": "lvot_peak_gradient",
    "lvot_vti": "lvot_vti",
    "pericardial_effusion": "pericardial_effusion", "clots_thrombus": "clots_thrombus",
    "conclusion_text": "impression_text",
}

# Execution priority. The output array is sorted by this, so the most urgent finding is the
# first card a clinician sees.
PRIORITY = {
    "Pediatric Review": 0,
    "Thrombus": 1, "Ischemia": 2, "Heart Failure": 3, "Pericardium": 4,
    "Cardiomyopathy": 5, "Hypertrophy": 6, "Pulmonary Pressure": 7, "Diastolic Function": 8,
    "Valves": 9, "Outflow Tract": 10, "Chambers": 11, "Aorta": 12,
    "Hypertensive Heart Disease": 13, "Right Heart": 14, "Conclusion-Only Findings": 15,
    "Athlete Screening": 16, "Exercise Safety": 17, "Risk Score": 18,
}

# ===========================================================================================
# Age bands
# ===========================================================================================
# Three bands, because the same measurement does not carry the same exercise risk at 25 and at
# 75. The bands shift thresholds; they never invent a finding, and an unknown age falls back to
# the standard (middle) numbers WITHOUT the band-specific extras -- guessing someone into a band
# would be exactly the kind of imputation G1 forbids.
AGE_BANDS = ("pediatric", "young", "middle", "older")


def resolve_age_band(raw_age) -> Optional[str]:
    """'73 years' / '73' / '73 / Female' -> 'older'; '6 months' -> 'pediatric'. None when no usable age was recorded.

    Reads the free-text patient_age the upload form collects, so no new field is needed.
    """
    age_years, is_unknown = V.parse_age_years(raw_age)
    if is_unknown or age_years is None:
        return None
    if age_years < 15.0:
        return "pediatric"
    if age_years < 40.0:
        return "young"
    if age_years <= 65.0:
        return "middle"
    return "older"


@dataclass(frozen=True)
class AgeThresholds:
    """The cut-offs one age band applies. Everything not listed here is age-independent."""
    label: str
    ef_contraindicated: float = 30.0
    ef_caution: float = 40.0
    pasp_contraindicated: float = 60.0
    pasp_caution: float = 40.0
    lvot_grad_contraindicated: float = 50.0
    lvot_grad_caution: float = 30.0
    lvot_vel_contraindicated: float = 3.5
    lvot_vel_caution: float = 2.5
    # None means "this band never contraindicates on septal thickness alone".
    ivsd_contraindicated: Optional[float] = None
    ivsd_caution: float = 15.0
    ischemia_suspicion: bool = False
    clear_advice: str = ""


_STANDARD_ADVICE = (
    "Exercise may be prescribed at usual guideline intensity, progressed as tolerated."
)

AGE_THRESHOLDS = {
    "pediatric": AgeThresholds(
        label="Pediatric (< 15 years)",
        clear_advice="Pediatric case -- numeric thresholds not clinically validated, mandatory manual specialist review required.",
    ),
    # Exercise-related sudden death from hypertrophic cardiomyopathy peaks in this band, so the
    # outflow-tract and septal-thickness screens are deliberately the strictest here.
    "young": AgeThresholds(
        label="Young (< 40 years)",
        lvot_grad_contraindicated=30.0, lvot_grad_caution=20.0,
        lvot_vel_contraindicated=2.7, lvot_vel_caution=2.0,
        ivsd_contraindicated=15.0, ivsd_caution=13.0,
        clear_advice="No structural contraindication in a patient under 40; a normal baseline "
                     "exercise tolerance can be assumed and full-intensity training progressed "
                     "as tolerated.",
    ),
    "middle": AgeThresholds(
        label="Middle-aged (40-65 years)",
        ischemia_suspicion=True,
        clear_advice=_STANDARD_ADVICE,
    ),
    # Older patients decompensate at milder values, so caution starts earlier on EF and PASP and
    # the default prescription is lower intensity even when nothing is flagged.
    "older": AgeThresholds(
        label="Older (> 65 years)",
        ef_caution=45.0, pasp_caution=35.0,
        clear_advice="No structural contraindication, but start at low intensity and progress "
                     "slowly: over-65 patients decompensate at milder values than the standard "
                     "thresholds capture.",
    ),
}

# Used when the report records no usable age. The standard numbers, without any band's extras.
_DEFAULT_THRESHOLDS = AgeThresholds(label="Age not recorded", clear_advice=_STANDARD_ADVICE)


def thresholds_for(band: Optional[str]) -> AgeThresholds:
    return AGE_THRESHOLDS.get(band or "", _DEFAULT_THRESHOLDS)


@dataclass
class Prediction:
    name: str
    category: str
    severity: str = "info"          # normal | mild | moderate | severe | info
    supporting_points: List[str] = field(default_factory=list)
    recommendation: str = ""
    # The parameter columns this finding actually rests on, by db field name. Named explicitly
    # rather than parsed back out of the evidence lines, because the three scores below have to
    # know WHICH parameter was measured to look up its age-resolved threshold (§9.2) and to see
    # whether a doctor filled it from a dropdown rather than the report (§9.1).
    fields: List[str] = field(default_factory=list)

    # --- §9 scores. Three INDEPENDENT 0-100 scales, not shares of one total. Only populated for
    # entries in the `diseases` list; Normal Heart, Athlete Screening, Exercise Safety and the
    # Risk Score leave them None, because "how far past threshold" is meaningless for a verdict
    # that fires on the ABSENCE of findings.
    confidence_score: Optional[int] = None
    prediction_score: Optional[int] = None
    severity_score: Optional[int] = None
    prediction_is_fallback: bool = False
    severity_is_fallback: bool = False

    @property
    def is_fallback_score(self) -> bool:
        return self.prediction_is_fallback or self.severity_is_fallback

    @property
    def priority(self) -> int:
        return PRIORITY.get(self.category, 99)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cardiac_disease_name": self.name,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "priority": self.priority,
            "supporting_points": self.supporting_points,
            "recommendation": self.recommendation,
            "fields": self.fields,
            "confidence_score": self.confidence_score,
            "prediction_score": self.prediction_score,
            "severity_score": self.severity_score,
            "prediction_is_fallback": self.prediction_is_fallback,
            "severity_is_fallback": self.severity_is_fallback,
            "is_fallback_score": self.is_fallback_score,
        }


class Params:
    """Normalized, unit-resolved view of one report. Every accessor returns None when absent."""

    def __init__(self, raw: Dict[str, Any], field_sources: Optional[Dict[str, str]] = None,
                 is_pediatric: bool = False):
        get = lambda spec: raw.get(PARAM_MAP.get(spec, spec)) if raw.get(PARAM_MAP.get(spec, spec)) is not None else raw.get(spec)  # noqa: E731
        self.raw = {spec: get(spec) for spec in PARAM_MAP}
        self.field_sources = field_sources or {}
        self.dropdown_fields = {f for f, src in self.field_sources.items() if src == "doctor_dropdown"}
        self.is_pediatric = is_pediatric

        self.ef = N.to_percent(self.raw["ejection_fraction"])
        self.lvidd = N.to_mm(self.raw["lvidd"])
        self.lvids = N.to_mm(self.raw["lvids"])
        self.ivsd = N.to_mm(self.raw["ivsd"])
        self.ivss = N.to_mm(self.raw["ivss"])
        self.pwd = N.to_mm(self.raw["pwd"])
        self.pws = N.to_mm(self.raw["pws"])

        meta = raw.get("extraction_meta") or {}
        ivsd_meta = meta.get("ivsd") or {}
        pwd_meta = meta.get("pwd") or {}
        self.ivsd_max = (
            N.to_mm(raw.get("ivsd_max"))
            or ivsd_meta.get("segment_values_mm", {}).get("max")
            or N.to_mm(raw.get("ivsd_apical"))
            or N.to_mm(raw.get("ivsd_mid"))
            or self.ivsd
        )
        self.pwd_max = (
            N.to_mm(raw.get("pwd_max"))
            or pwd_meta.get("segment_values_mm", {}).get("max")
            or N.to_mm(raw.get("pwd_apical"))
            or N.to_mm(raw.get("pwd_mid"))
            or self.pwd
        )
        self.la = N.to_mm(self.raw["la_diameter"])
        self.la_bsa = N.to_index(self.raw["la_diameter_bsa"])
        self.rwt = N.to_ratio(self.raw["rwt"])
        self.lv_mass_index = N.to_index(self.raw["lv_mass_index"])
        self.aortic_root = N.to_mm(self.raw["aortic_root"])
        self.aortic_annulus = N.to_mm(self.raw["aortic_annulus"])
        self.stj = N.to_mm(self.raw["stj"])
        self.aortic_diameter = N.to_mm(self.raw["aortic_diameter"])
        self.ea = N.to_ratio(self.raw["ea_ratio"])
        self.pasp = N.to_gradient_mmhg(self.raw["pasp"])
        self.av_vel = N.to_velocity_ms(self.raw["av_peak_velocity"])
        self.av_grad = N.to_gradient_mmhg(self.raw["av_peak_gradient"])
        self.mv_vel = N.to_velocity_ms(self.raw["mv_peak_velocity"])
        self.mv_grad = N.to_gradient_mmhg(self.raw["mv_peak_gradient"])
        self.tv_vel = N.to_velocity_ms(self.raw["tv_peak_velocity"])
        self.tv_grad = N.to_gradient_mmhg(self.raw["tv_peak_gradient"])
        self.pv_vel = N.to_velocity_ms(self.raw["pv_peak_velocity"])
        self.pv_grad = N.to_gradient_mmhg(self.raw["pv_peak_gradient"])
        self.lvot_vel = N.to_velocity_ms(self.raw["lvot_peak_velocity"])
        self.lvot_grad = N.to_gradient_mmhg(self.raw["lvot_peak_gradient"])

        self.conclusion = N.clean(self.raw["conclusion_text"]) or ""
        self.ra_size = self.raw["ra_size"]
        self.rv_size = self.raw["rv_size"]
        self.ivc = self.raw["ivc"]
        self.effusion = self.raw["pericardial_effusion"]
        self.thrombus = self.raw["clots_thrombus"]

        self.valves = {
            "Aortic": self.raw["aortic_valve_finding"],
            "Mitral": self.raw["mitral_valve_finding"],
            "Tricuspid": self.raw["tricuspid_valve_finding"],
            "Pulmonary": self.raw["pulmonary_valve_finding"],
        }
        self.regional_walls = {
            "Septal": self.raw["septal_wall_motion"], "Anterior": self.raw["anterior_wall_motion"],
            "Inferior": self.raw["inferior_wall_motion"], "Lateral": self.raw["lateral_wall_motion"],
            "Posterior": self.raw["posterior_wall_motion"], "Apical": self.raw["apical_wall_motion"],
        }

        # Parsed numbers keyed by DB FIELD NAME -- the same names value_table.py and the review
        # page use. The scoring layer (§9) looks a finding's parameter up here to measure how far
        # past its age-resolved threshold the value sat.
        self.numbers: Dict[str, Optional[float]] = {
            "ef": self.ef, "lvidd": self.lvidd, "lvids": self.lvids,
            "ivsd": self.ivsd, "ivss": self.ivss, "pwd": self.pwd, "pws": self.pws,
            "la_diameter": self.la, "la_diameter_indexed_value": self.la_bsa,
            "relative_wall_thickness": self.rwt, "lv_mass": self.lv_mass_index,
            "ao_root": self.aortic_root, "ao_annulus": self.aortic_annulus,
            "ao_stj": self.stj, "ao_diameter": self.aortic_diameter,
            "e_a_ratio": self.ea, "pasp": self.pasp,
            "av_peak_velocity": self.av_vel, "av_peak_gradient": self.av_grad,
            "mv_peak_velocity": self.mv_vel, "mv_peak_gradient": self.mv_grad,
            "tv_peak_velocity": self.tv_vel, "tv_peak_gradient": self.tv_grad,
            "pv_peak_velocity": self.pv_vel, "pv_peak_gradient": self.pv_grad,
            "lvot_peak_velocity": self.lvot_vel, "lvot_peak_gradient": self.lvot_grad,
        }

    @property
    def ivsd_thresh(self) -> Optional[float]:
        """Max-aware IVSd thickness across basal, mid, apical segments."""
        vals = [self.ivsd, getattr(self, "ivsd_max", None)]
        filtered = [v for v in vals if v is not None]
        return max(filtered) if filtered else None

    @property
    def pwd_thresh(self) -> Optional[float]:
        """Max-aware PWd thickness across basal, mid, apical segments."""
        vals = [self.pwd, getattr(self, "pwd_max", None)]
        filtered = [v for v in vals if v is not None]
        return max(filtered) if filtered else None

    def is_dropdown(self, field: str) -> bool:
        """True if the field is a fabricated placeholder from doctor_dropdown that has not been measured/confirmed."""
        db_f = PARAM_MAP.get(field, field)
        return (field in self.dropdown_fields) or (db_f in self.dropdown_fields)

    def says(self, *phrases: str) -> bool:
        """G3 -- conclusion_text substring check (non-negated matches only)."""
        return N.contains(self.conclusion, *phrases)

    def quote(self, *phrases: str) -> str:
        """The matched phrase AS THE REPORT WROTE IT, for the supporting point.

        Slices the original conclusion rather than echoing the search token, prioritizing
        non-negated matches. Longest phrase first, so a spelled-out match is preferred
        over its abbreviation.
        """
        if not self.conclusion:
            return phrases[0] if phrases else ""
        for p in sorted(phrases, key=len, reverse=True):
            p_clean = p.strip()
            if not p_clean:
                continue
            is_word = len(p_clean) <= 3
            matches = N.find_matches(self.conclusion, p_clean, is_word_boundary=is_word)
            for start, end in matches:
                if not N.is_negated_match(self.conclusion, start, end):
                    return self.conclusion[start:end]
            if matches:
                return self.conclusion[matches[0][0]:matches[0][1]]
        return phrases[0] if phrases else ""


def _from_conclusion(p: Params, *phrases: str) -> str:
    return f"Conclusion text explicitly mentioned '{p.quote(*phrases)}'"


def _severity_near(conclusion: str, *phrases: str) -> Optional[str]:
    """Severity word qualifying THIS phrase, not the worst word anywhere in the conclusion.

    A conclusion routinely grades several things at once -- "Moderate LV Dysfunction. Mild MR."
    Scanning the whole string would report the mitral regurgitation as Moderate, overstating it.
    Only the text immediately preceding the phrase is considered, prioritizing non-negated matches.
    """
    text = N.clean(conclusion)
    if not text:
        return None
    low = text.lower()
    for phrase in phrases:
        p_clean = phrase.strip()
        if not p_clean:
            continue
        is_word = len(p_clean) <= 3
        matches = N.find_matches(text, p_clean, is_word_boundary=is_word)
        non_neg = [start for start, end in matches if not N.is_negated_match(text, start, end)]
        for idx in (non_neg or [m[0] for m in matches]):
            window = low[max(0, idx - 28):idx]
            for word in reversed(N.SEVERITY_ORDER):
                if word in window:
                    return word.title()
    return None


# ===========================================================================================
# Compulsory-group grading
# ===========================================================================================
# The nine compulsory groups are the parameters the review page now insists on, so they are the
# ones most likely to actually be present. Grading them from NUMBERS here -- and reading the
# same numbers from both the disease rules and the exercise screen -- means a valve severity is
# computed from the measured gradient rather than inferred from whatever adjective the report
# happened to print, and that the two halves of the engine can never disagree about it.
#
# Bands are the standard severe/moderate ones expressed as PEAK values (peak = 4v^2), so the
# gradient rung and the velocity rung of each chain grade the same lesion identically.
def _as_grade(grad: Optional[float], vel: Optional[float]) -> Tuple[Optional[str], str]:
    """Aortic stenosis severity from the compulsory chain: peak gradient, then peak velocity."""
    if grad is not None:
        if grad >= 64:
            return "Severe", f"AV peak gradient is {grad:g} mmHg (Threshold: >= 64 mmHg)"
        if grad >= 36:
            return "Moderate", f"AV peak gradient is {grad:g} mmHg (Threshold: 36-63 mmHg)"
        if grad >= 20:
            return "Mild", f"AV peak gradient is {grad:g} mmHg (Threshold: 20-35 mmHg)"
        return None, f"AV peak gradient is {grad:g} mmHg (< 20 mmHg)"
    if vel is not None:
        if vel >= 4.0:
            return "Severe", f"AV peak velocity is {vel:g} m/s (Threshold: >= 4.0 m/s)"
        if vel >= 3.0:
            return "Moderate", f"AV peak velocity is {vel:g} m/s (Threshold: 3.0-3.9 m/s)"
        if vel > 2.5:
            return "Mild", f"AV peak velocity is {vel:g} m/s (Threshold: > 2.5 m/s)"
        return None, f"AV peak velocity is {vel:g} m/s (<= 2.5 m/s)"
    return None, ""


def _ms_grade(grad: Optional[float], vel: Optional[float]) -> Tuple[Optional[str], str]:
    """Mitral stenosis severity from the compulsory chain: peak gradient, then peak velocity.

    Conservative on purpose: mitral stenosis is graded clinically on the MEAN gradient, and a
    peak always exceeds its mean, so these bands flag early rather than late.
    """
    if grad is not None:
        if grad >= 20:
            return "Severe", f"MV peak gradient is {grad:g} mmHg (Threshold: >= 20 mmHg peak)"
        if grad >= 10:
            return "Moderate", f"MV peak gradient is {grad:g} mmHg (Threshold: 10-19 mmHg peak)"
        if grad >= 5:
            return "Mild", f"MV peak gradient is {grad:g} mmHg (Threshold: 5-9 mmHg peak)"
        return None, f"MV peak gradient is {grad:g} mmHg (< 5 mmHg)"
    if vel is not None:
        if vel >= 2.2:
            return "Severe", f"MV peak velocity is {vel:g} m/s (Threshold: >= 2.2 m/s)"
        if vel >= 1.6:
            return "Moderate", f"MV peak velocity is {vel:g} m/s (Threshold: 1.6-2.1 m/s)"
        if vel >= 1.3:
            return "Mild", f"MV peak velocity is {vel:g} m/s (Threshold: 1.3-1.5 m/s)"
        return None, f"MV peak velocity is {vel:g} m/s (< 1.3 m/s)"
    return None, ""


# The nine groups, and the fields each one falls back through. Single source of truth for both
# _rule_exercise_safety and the coverage report, and the list the review page's "Compulsory
# Parameters" section mirrors field for field.
COMPULSORY_GROUPS = (
    ("Ejection Fraction", ("ejection_fraction",)),
    ("Aortic Stenosis", ("av_peak_gradient", "av_peak_velocity", "aortic_valve_finding")),
    ("Mitral Stenosis", ("mv_peak_gradient", "mv_peak_velocity", "mitral_valve_finding")),
    ("Pulmonary Pressure (PASP)", ("pasp", "tv_peak_gradient", "tv_peak_velocity", "ivc")),
    ("Pericardial Effusion", ("pericardial_effusion",)),
    ("Clots / Thrombus", ("clots_thrombus",)),
    ("LVOT Obstruction", ("lvot_peak_gradient", "lvot_peak_velocity")),
    ("Regional Wall Motion Abnormality (Ischemia)", ("rwma", "wall_motion")),
    ("Septal Thickness (HOCM screening)", ("ivsd",)),
)


def _compulsory_coverage(p: Params) -> Dict[str, Any]:
    """Which of the nine groups this report can actually be graded on.

    Reported alongside the predictions so a clinician can see how much of the engine's basis was
    present, rather than having to infer it from which cards happen to be missing.
    """
    present, missing = [], []
    for name, fields in COMPULSORY_GROUPS:
        (present if any(N.clean(p.raw.get(f)) is not None for f in fields) else missing).append(name)
    return {
        "groups_total": len(COMPULSORY_GROUPS),
        "groups_present": len(present),
        "present": present,
        "missing": missing,
    }


# ===========================================================================================
# Rule groups
# ===========================================================================================
def _rule_thrombus(p: Params) -> List[Prediction]:
    if N.contains(p.thrombus, "present", "detected", "seen", "thrombus", "clot") \
            and not N.is_absent(p.thrombus):
        return [Prediction("Intracardiac Thrombus", "Thrombus", "severe",
                           [f"Clots / Thrombus reported as '{N.clean(p.thrombus)}'"],
                           "Urgent review -- anticoagulation and embolic risk assessment.",
                           fields=["clots_thrombus"])]
    if p.says("thrombus", "intracardiac clot"):
        return [Prediction("Intracardiac Thrombus", "Thrombus", "severe",
                           [_from_conclusion(p, "Thrombus", "Intracardiac Clot")],
                           "Urgent review -- anticoagulation and embolic risk assessment.")]
    return []


# Region label -> the db field it was read from, so a positive RWMA can name its own sources.
_WALL_FIELDS = {
    "Septal": "septal_wall_motion", "Anterior": "anterior_wall_motion",
    "Inferior": "inferior_wall_motion", "Lateral": "lateral_wall_motion",
    "Posterior": "posterior_wall_motion", "Apical": "apical_wall_motion",
}


def _rwma_present(p: Params):
    """(present, points, fields). present is None when there is no information either way."""
    points, fields, known = [], [], False
    if N.clean(p.raw["rwma"]) is not None:
        known = True
        if N.contains(p.raw["rwma"], "yes", "present") and not N.is_absent(p.raw["rwma"]):
            points.append(f"RWMA reported as '{N.clean(p.raw['rwma'])}'")
            fields.append("rwma")
    for region, value in p.regional_walls.items():
        fname = _WALL_FIELDS[region]
        kw = N.keyword(value)
        if kw is None:
            continue
        known = True
        if any(w in kw for w in N.ABNORMAL_MOTION_WORDS):
            points.append(f"{region} wall motion is '{N.clean(value)}'")
            fields.append(fname)
    if not known:
        return None, points, fields
    return bool(points), points, fields


def _rule_ischemia(p: Params, t: AgeThresholds = _DEFAULT_THRESHOLDS) -> List[Prediction]:
    out: List[Prediction] = []
    present, rwma_points, rwma_fields = _rwma_present(p)
    if present:
        out.append(Prediction("Regional Wall Motion Abnormality (RWMA)", "Ischemia", "moderate",
                              rwma_points,
                              "Correlate with coronary anatomy / ECG for acute or previous ischemia.",
                              fields=rwma_fields))
    elif p.says("rwma", "regional wall motion"):
        out.append(Prediction("Regional Wall Motion Abnormality (RWMA)", "Ischemia", "moderate",
                              [_from_conclusion(p, "RWMA", "Regional Wall Motion")],
                              "Correlate with coronary anatomy / ECG for acute or previous ischemia."))

    cad_points: List[str] = []
    cad_fields: List[str] = list(rwma_fields)
    if present:
        cad_points.append("Regional wall motion abnormality present")
    if not p.is_pediatric:
        ef_cut = 50.0 if t.ischemia_suspicion else 45.0
        if (p.ef is not None
                and p.lvids is not None
                and p.ef < ef_cut and p.lvids > 40):
            cad_points.append(f"EF is {p.ef:g}% (< {ef_cut:g}%) with LVIDs {p.lvids:g} mm (> 40 mm)")
            cad_fields.extend(("ef", "lvids"))
            if t.ischemia_suspicion:
                cad_points.append(f"{t.label}: EF threshold widened from 45% to 50% -- baseline "
                                  f"suspicion for ischemic disease is higher in this age band")
    if p.says("cad", "ihd", "ischemic heart disease", "old mi", "previous infarction"):
        cad_points.append(_from_conclusion(p, "CAD", "IHD", "Ischemic Heart Disease", "Old MI",
                                           "Previous Infarction"))
    if cad_points:
        out.append(Prediction("Possible Ischemic Heart Disease / CAD", "Ischemia", "moderate",
                              cad_points,
                              "PROXY FINDING ONLY -- a 2D Echo cannot demonstrate coronary "
                              "blockage. Recommend ECG / stress test / angiography as indicated.",
                              fields=cad_fields))
    return out


def _rule_lv_function(p: Params) -> List[Prediction]:
    out: List[Prediction] = []
    if p.ef is not None and not p.is_pediatric:
        if p.ef < 35:
            out.append(Prediction("Severe Left Ventricular Dysfunction", "Heart Failure", "severe",
                                  [f"EF is {p.ef:g}% (Threshold: < 35%)"],
                                  "Severe systolic impairment -- guideline-directed heart failure "
                                  "therapy and prompt cardiology review.", fields=["ef"]))
        elif p.ef < 45:
            out.append(Prediction("Moderate Left Ventricular Dysfunction", "Heart Failure",
                                  "moderate", [f"EF is {p.ef:g}% (Threshold: 35-44%)"],
                                  "Moderate systolic impairment -- optimise medical therapy.",
                                  fields=["ef"]))
        elif p.ef < 55:
            out.append(Prediction("Mild Left Ventricular Dysfunction", "Heart Failure", "mild",
                                  [f"EF is {p.ef:g}% (Threshold: 45-54%)"],
                                  "Mild systolic impairment -- monitor and address risk factors.",
                                  fields=["ef"]))
    else:
        # Pattern-general conclusion matching for LV systolic function across phrasing variants
        _SEV_LV_PHRASES = (
            "severe lv dysfunction", "severe lv systolic dysfunction", "severely depressed lv systolic function",
            "severely reduced lv systolic function", "severely impaired lv systolic function", "severe systolic dysfunction",
            "severely depressed lvef", "poor lv systolic function", "grossly depressed lv systolic function",
            "severe left ventricular dysfunction", "severe left ventricular systolic dysfunction"
        )
        _MOD_LV_PHRASES = (
            "moderate lv dysfunction", "moderate lv systolic dysfunction", "moderately depressed lv systolic function",
            "moderately reduced lv systolic function", "moderately impaired lv systolic function", "moderate systolic dysfunction",
            "moderately depressed lvef", "fair lv systolic function", "moderate left ventricular dysfunction",
            "moderate left ventricular systolic dysfunction"
        )
        _MILD_LV_PHRASES = (
            "mild lv dysfunction", "mild lv systolic dysfunction", "mildly depressed lv systolic function",
            "mildly reduced lv systolic function", "mildly impaired lv systolic function", "mild systolic dysfunction",
            "mildly depressed lvef", "mild left ventricular dysfunction", "mild left ventricular systolic dysfunction"
        )
        _GEN_LV_PHRASES = (
            "lv dysfunction", "lv systolic dysfunction", "depressed lv systolic function",
            "reduced lv systolic function", "impaired lv systolic function", "systolic dysfunction", "depressed lvef"
        )

        if p.says(*_SEV_LV_PHRASES):
            out.append(Prediction("Severe Left Ventricular Dysfunction", "Heart Failure", "severe",
                                  [_from_conclusion(p, *_SEV_LV_PHRASES)],
                                  "Severe systolic impairment stated in report conclusion."))
        elif p.says(*_MOD_LV_PHRASES):
            out.append(Prediction("Moderate Left Ventricular Dysfunction", "Heart Failure", "moderate",
                                  [_from_conclusion(p, *_MOD_LV_PHRASES)],
                                  "Moderate systolic impairment stated in report conclusion."))
        elif p.says(*_MILD_LV_PHRASES):
            out.append(Prediction("Mild Left Ventricular Dysfunction", "Heart Failure", "mild",
                                  [_from_conclusion(p, *_MILD_LV_PHRASES)],
                                  "Mild systolic impairment stated in report conclusion."))
        elif p.says(*_GEN_LV_PHRASES) and not p.says("preserved lv systolic", "good lv systolic", "normal lv systolic", "normal systolic"):
            out.append(Prediction("Left Ventricular Dysfunction", "Heart Failure", "moderate",
                                  [_from_conclusion(p, *_GEN_LV_PHRASES)],
                                  "Systolic impairment stated in report conclusion."))

    hf_points: List[str] = []
    hf_fields: List[str] = []
    if p.ef is not None and not p.is_pediatric and p.ef < 40:
        hf_points.append(f"EF is {p.ef:g}% (Threshold: < 40%)")
        hf_fields.append("ef")
    if p.says("heart failure", "hfref", "severe lv dysfunction", "congestive heart failure", "chf"):
        hf_points.append(_from_conclusion(p, "Heart Failure", "HFrEF", "Severe LV Dysfunction", "CHF", "Congestive Heart Failure"))
    if hf_points:
        out.append(Prediction("Heart Failure with Reduced Ejection Fraction (HFrEF)",
                              "Heart Failure", "severe", hf_points,
                              "Initiate/optimise guideline-directed heart failure therapy.",
                              fields=hf_fields))
    return out


def _rule_pericardium(p: Params) -> List[Prediction]:
    grade = N.severity_of(p.effusion) or (
        "Large" if N.contains(p.effusion, "large", "massive") else (
            "Small" if N.contains(p.effusion, "small") else None
        )
    )
    has_effusion = (N.clean(p.effusion) is not None and not N.is_absent(p.effusion)
                    and N.contains(p.effusion, "effusion", "mild", "small", "moderate", "large",
                                   "massive", "severe", "present"))
    if not has_effusion and p.says("pericardial effusion"):
        has_effusion, grade = True, N.severity_of(p.conclusion)
    if not has_effusion:
        return []
    sev = {"Small": "mild", "Mild": "mild", "Moderate": "moderate"}.get(grade or "", "severe")
    label = f"Pericardial Effusion ({grade})" if grade else "Pericardial Effusion"
    return [Prediction(label, "Pericardium", sev,
                       [f"Pericardial effusion reported as '{N.clean(p.effusion) or p.quote('Pericardial Effusion')}'"],
                       "Assess size and haemodynamic significance; exclude tamponade physiology.",
                       fields=["pericardial_effusion"] if N.clean(p.effusion) else [])]


def _rule_cardiomyopathy(p: Params) -> List[Prediction]:
    out: List[Prediction] = []
    if (not p.is_pediatric and p.ef is not None
            and p.lvidd is not None
            and p.lvids is not None
            and p.ef < 45 and p.lvidd > 58 and p.lvids > 40):
        out.append(Prediction("Dilated Cardiomyopathy (DCM)", "Cardiomyopathy", "severe",
                              [f"EF is {p.ef:g}% (Threshold: < 45%)",
                               f"LVIDd is {p.lvidd:g} mm (Threshold: > 58 mm)",
                               f"LVIDs is {p.lvids:g} mm (Threshold: > 40 mm)"],
                              "Dilated phenotype -- heart failure therapy and aetiology workup.",
                              fields=["ef", "lvidd", "lvids"]))
    elif p.says("dilated cardiomyopathy", "dcm", "dilated phenotype"):
        out.append(Prediction("Dilated Cardiomyopathy (DCM)", "Cardiomyopathy", "severe",
                              [_from_conclusion(p, "Dilated Cardiomyopathy", "DCM", "Dilated Phenotype")],
                              "Dilated phenotype stated in the report conclusion."))
    return out


def _rule_hypertrophy(p: Params, t: AgeThresholds = _DEFAULT_THRESHOLDS) -> List[Prediction]:
    out: List[Prediction] = []
    points: List[str] = []
    lvh_fields: List[str] = []
    ivsd_thresh = p.ivsd_thresh
    pwd_thresh = p.pwd_thresh

    if not p.is_pediatric:
        if ivsd_thresh is not None and ivsd_thresh > 11:
            points.append(f"IVSd is {ivsd_thresh:g} mm (Threshold: > 11 mm)")
            lvh_fields.append("ivsd")
        if pwd_thresh is not None and pwd_thresh > 11:
            points.append(f"PWd is {pwd_thresh:g} mm (Threshold: > 11 mm)")
            lvh_fields.append("pwd")
        if p.lv_mass_index is not None and p.lv_mass_index > 115:
            points.append(f"LV Mass Index is {p.lv_mass_index:g} g/m2 (Threshold: > 115 g/m2)")
            lvh_fields.append("lv_mass")

    for label, value, fname in (("IVSd", p.raw["ivsd"], "ivsd"), ("PWd", p.raw["pwd"], "pwd")):
        if N.contains(value, "thicken"):
            points.append(f"{label} reported as '{N.clean(value)}'")
            if fname not in lvh_fields:
                lvh_fields.append(fname)
    if p.says("lvh", "left ventricular hypertrophy"):
        points.append(_from_conclusion(p, "LVH", "Left Ventricular Hypertrophy"))

    if points:
        severe = ((ivsd_thresh is not None and ivsd_thresh >= 15) or
                  (pwd_thresh is not None and pwd_thresh >= 15))
        if severe and not p.is_pediatric:
            points.append("Wall thickness >= 15 mm indicates severe hypertrophy")
        out.append(Prediction("Severe Left Ventricular Hypertrophy" if (severe and not p.is_pediatric)
                              else "Left Ventricular Hypertrophy (LVH)",
                              "Hypertrophy", "severe" if (severe and not p.is_pediatric) else "moderate", points,
                              "Assess for hypertension, aortic stenosis or infiltrative disease.",
                              fields=lvh_fields))

    # LV geometry
    rwt_val = p.rwt
    if rwt_val is None and pwd_thresh is not None and p.lvidd is not None and p.lvidd > 0:
        rwt_val = round(2.0 * pwd_thresh / p.lvidd, 2)

    if not p.is_pediatric and rwt_val is not None:
        mass_high = p.lv_mass_index is not None and p.lv_mass_index > 115
        septum_high = ivsd_thresh is not None and ivsd_thresh > 11
        if rwt_val > 0.42 and (mass_high or septum_high):
            geo = [f"RWT is {rwt_val:g} (Threshold: > 0.42)"]
            geo_fields = ["relative_wall_thickness"]
            if mass_high:
                geo.append(f"LV Mass Index is {p.lv_mass_index:g} g/m2 (Threshold: > 115 g/m2)")
                geo_fields.append("lv_mass")
            if septum_high:
                geo.append(f"IVSd is {ivsd_thresh:g} mm (Threshold: > 11 mm)")
                geo_fields.append("ivsd")
            out.append(Prediction("Concentric Left Ventricular Hypertrophy", "Hypertrophy",
                                  "moderate", geo, "Concentric hypertrophic remodeling pattern.",
                                  fields=geo_fields))
        elif (rwt_val > 0.42 and p.lv_mass_index is not None and p.lv_mass_index <= 115
                and ivsd_thresh is not None and ivsd_thresh <= 11):
            out.append(Prediction("Concentric Remodeling", "Hypertrophy", "mild",
                                  [f"RWT is {rwt_val:g} (Threshold: > 0.42)",
                                   f"LV Mass Index is normal at {p.lv_mass_index:g} g/m2",
                                   f"IVSd is {ivsd_thresh:g} mm (<= 11 mm)"],
                                  "Remodeling without hypertrophy -- address afterload/blood pressure.",
                                  fields=["relative_wall_thickness"]))
    elif p.says("concentric lvh", "concentric lv hypertrophy", "concentric hypertrophy", "concentric left ventricular hypertrophy"):
        out.append(Prediction("Concentric Left Ventricular Hypertrophy", "Hypertrophy", "moderate",
                              [_from_conclusion(p, "Concentric LVH", "Concentric Hypertrophy")],
                              "Concentric hypertrophic remodeling pattern."))

    # HCM guardrail & subtype matching (Apical HCM, Asymmetric Septal Hypertrophy / ASH, HOCM)
    if p.says("apical hcm", "apical hypertrophic cardiomyopathy", "apical hypertrophy", "hypertrophic cardiomyopathy (apical hcm)", "apical type hcm"):
        out.append(Prediction("Hypertrophic Cardiomyopathy", "Hypertrophy", "severe",
                              [_from_conclusion(p, "Apical HCM", "Apical Hypertrophic Cardiomyopathy", "Apical Hypertrophy")],
                              "Stated in the report conclusion (Apical HCM) -- specialist cardiomyopathy review."))
    elif p.says("hypertrophic cardiomyopathy", "hcm", "hocm", "hypertrophic obstructive cardiomyopathy", "asymmetric septal hypertrophy", "ash"):
        out.append(Prediction("Hypertrophic Cardiomyopathy", "Hypertrophy", "severe",
                              [_from_conclusion(p, "Hypertrophic Cardiomyopathy", "HCM", "HOCM", "Asymmetric Septal Hypertrophy")],
                              "Stated in the report conclusion -- specialist cardiomyopathy review."))
    elif (not p.is_pediatric and ivsd_thresh is not None
            and p.ef is not None
            and ivsd_thresh >= t.ivsd_caution and p.ef >= 55):
        hcm_points = [f"IVSd is {ivsd_thresh:g} mm (Threshold: >= {t.ivsd_caution:g} mm)",
                      f"EF is {p.ef:g}% (Threshold: >= 55%)",
                      "Reported as POSSIBLE only -- HCM is never diagnosed from "
                      "measurements alone"]
        if t.ivsd_caution < 15.0:
            hcm_points.insert(1, f"{t.label}: screening cut-off lowered from 15 mm to "
                                 f"{t.ivsd_caution:g} mm for this age band")
        out.append(Prediction("Possible Hypertrophic Cardiomyopathy", "Hypertrophy", "moderate",
                              hcm_points,
                              "Suggestive pattern only. Requires specialist assessment; do not "
                              "treat as a diagnosis.", fields=["ivsd", "ef"]))
    return out


def _rule_pulmonary_pressure(p: Params) -> List[Prediction]:
    points, grade, ph_fields = [], None, []
    if not p.is_pediatric and p.pasp is not None and p.pasp > 35:
        grade = "Mild" if p.pasp <= 45 else ("Moderate" if p.pasp <= 60 else "Severe")
        points.append(f"PASP is {p.pasp:g} mmHg (Threshold: > 35 mmHg -- {grade} range)")
        ph_fields.append("pasp")
    if N.keyword(p.ivc) == "raised right atrial pressure" or N.contains(p.ivc, "plethoric"):
        points.append(f"IVC reported as '{N.clean(p.ivc)}' (raised right atrial pressure)")
        ph_fields.append("ivc")
    if p.says("pulmonary hypertension", "pulmonary arterial hypertension", "pah", "pulmonary venous hypertension", "ph"):
        points.append(_from_conclusion(p, "Pulmonary Hypertension", "PAH", "PH"))
        if not grade:
            grade = _severity_near(p.conclusion, "pulmonary hypertension", "pah", "ph")
    if not points:
        return []
    label = f"Pulmonary Hypertension ({grade})" if grade else "Pulmonary Hypertension"
    return [Prediction(label, "Pulmonary Pressure",
                       {"Mild": "mild", "Moderate": "moderate"}.get(grade or "", "severe"),
                       points, "Evaluate left heart disease, lung disease and thromboembolism.",
                       fields=ph_fields)]


def _rule_diastolic(p: Params) -> List[Prediction]:
    # Conclusion / text-sourced diastolic dysfunction matching (takes precedence as physician's definitive clinical impression)
    _G3_PHRASES = (
        "grade iii diastolic", "grade 3 diastolic", "grade three diastolic",
        "grade iii lv diastolic", "grade 3 lv diastolic", "grade iii diastolic lv",
        "grade 3 diastolic lv", "grade 3+ diastolic", "grade iii+ diastolic",
        "restrictive filling", "advanced diastolic dysfunction", "severe diastolic dysfunction",
        "diastolic dysfunction grade iii", "diastolic dysfunction grade 3", "diastolic dysfunction grade three",
        "diastolic lv dysfunction grade iii", "diastolic lv dysfunction grade 3", "diastolic lv dysfunction grade three",
        "diastolic dysfunction (grade iii)", "diastolic dysfunction (grade 3)", "diastolic dysfunction - grade iii", "diastolic dysfunction - grade 3",
        "grade iii dd", "grade 3 dd", "grade 3+ dd", "grade iii+ dd",
    )
    _G2_PHRASES = (
        "grade ii diastolic", "grade 2 diastolic", "grade two diastolic",
        "grade ii lv diastolic", "grade 2 lv diastolic", "grade ii diastolic lv",
        "grade 2 diastolic lv", "grade 2+ diastolic", "grade ii+ diastolic",
        "pseudonormal", "moderate diastolic dysfunction",
        "diastolic dysfunction grade ii", "diastolic dysfunction grade 2", "diastolic dysfunction grade two",
        "diastolic lv dysfunction grade ii", "diastolic lv dysfunction grade 2", "diastolic lv dysfunction grade two",
        "diastolic dysfunction (grade ii)", "diastolic dysfunction (grade 2)", "diastolic dysfunction - grade ii", "diastolic dysfunction - grade 2",
        "grade ii dd", "grade 2 dd", "grade 2+ dd", "grade ii+ dd",
    )
    _G1_PHRASES = (
        "grade i diastolic", "grade 1 diastolic", "grade one diastolic",
        "grade i lv diastolic", "grade 1 lv diastolic", "grade i diastolic lv",
        "grade 1 diastolic lv", "grade 1+ diastolic", "grade i+ diastolic",
        "impaired relaxation", "mild diastolic dysfunction",
        "diastolic dysfunction grade i", "diastolic dysfunction grade 1", "diastolic dysfunction grade one",
        "diastolic lv dysfunction grade i", "diastolic lv dysfunction grade 1", "diastolic lv dysfunction grade one",
        "diastolic dysfunction (grade i)", "diastolic dysfunction (grade 1)", "diastolic dysfunction - grade i", "diastolic dysfunction - grade 1",
        "grade i dd", "grade 1 dd", "grade 1+ dd", "grade i+ dd",
    )
    _GEN_PHRASES = (
        "diastolic dysfunction", "diastolic lv dysfunction", "lv diastolic dysfunction",
        "diastolic function impaired",
    )

    if p.says(*_G3_PHRASES) or (p.says("grade iii", "grade 3", "grade three", "restrictive") and p.says("diastolic", "lvd", "dysfunction", "filling", "dd")):
        return [Prediction("Grade III Diastolic Dysfunction (Restrictive Filling)",
                           "Diastolic Function", "severe",
                           [_from_conclusion(p, *_G3_PHRASES, "Grade III", "Grade 3", "Restrictive")],
                           "Restrictive filling stated in the report conclusion.")]

    if p.says(*_G2_PHRASES) or (p.says("grade ii", "grade 2", "grade two", "pseudonormal") and p.says("diastolic", "lvd", "dysfunction", "filling", "dd")):
        return [Prediction("Grade II Diastolic Dysfunction (Pseudonormal)",
                           "Diastolic Function", "moderate",
                           [_from_conclusion(p, *_G2_PHRASES, "Grade II", "Grade 2", "Pseudonormal")],
                           "Pseudonormal diastolic filling pattern stated in the report conclusion.")]

    if p.says(*_G1_PHRASES) or (p.says("grade i", "grade 1", "grade one", "impaired relaxation") and p.says("diastolic", "lvd", "dysfunction", "filling", "dd")):
        return [Prediction("Grade I Diastolic Dysfunction (Impaired Relaxation)",
                           "Diastolic Function", "mild",
                           [_from_conclusion(p, *_G1_PHRASES, "Grade I", "Grade 1", "Impaired Relaxation")],
                           "Impaired relaxation stated in the report conclusion -- common with age and hypertension.")]

    if p.says(*_GEN_PHRASES):
        return [Prediction("Diastolic Dysfunction",
                           "Diastolic Function", "mild",
                           [_from_conclusion(p, *_GEN_PHRASES)],
                           "Diastolic dysfunction stated in the report conclusion.")]

    # Measurement-based calculation when no explicit conclusion text matched
    if p.ea is not None:
        if p.ea < 0.8:
            return [Prediction("Grade I Diastolic Dysfunction (Impaired Relaxation)",
                               "Diastolic Function", "mild",
                               [f"E/A ratio is {p.ea:g} (Threshold: < 0.8)"],
                               "Impaired relaxation -- common with age and hypertension.",
                               fields=["e_a_ratio"])]
        if p.ea > 2.0:
            return [Prediction("Grade III Diastolic Dysfunction (Restrictive Filling)",
                               "Diastolic Function", "severe",
                               [f"E/A ratio is {p.ea:g} (Threshold: > 2.0)"],
                               "Restrictive filling -- elevated filling pressures.",
                               fields=["e_a_ratio"])]
        if p.la is not None and p.la > 40:
            return [Prediction("Grade II Diastolic Dysfunction (Pseudonormal)",
                               "Diastolic Function", "moderate",
                               [f"E/A ratio is {p.ea:g} (Threshold: 0.8-2.0)",
                                f"LA diameter is {p.la:g} mm (Threshold: > 40 mm)"],
                               "Pseudonormal filling with left atrial enlargement.",
                               fields=["e_a_ratio", "la_diameter"])]

    return []


_REGURG_TOKENS = {"Aortic": "ar", "Mitral": "mr", "Tricuspid": "tr", "Pulmonary": "pr"}
_STENOSIS_TOKENS = {"Aortic": "as", "Mitral": "ms", "Tricuspid": "ts", "Pulmonary": "ps"}

# Valve -> the db fields that valve's findings are read from, for Prediction.fields.
_VALVE_FIELDS = {
    "Aortic": ("av_finding", "av_peak_velocity", "av_peak_gradient"),
    "Mitral": ("mv_finding", "mv_peak_velocity", "mv_peak_gradient"),
    "Tricuspid": ("tv_finding", "tv_peak_velocity", "tv_peak_gradient"),
    "Pulmonary": ("pv_finding", "pv_peak_velocity", "pv_peak_gradient"),
}

_VALVE_NORMAL_PHRASES = {
    "Aortic": (
        "normal aortic", "normal, trileaflet aortic", "normal trileaflet aortic",
        "aortic valve is normal", "aortic valve normal", "aortic valve: normal",
        "normal av", "av is normal", "av normal", "av: normal",
        "aortic valve unremarkable", "aortic valve wnl",
        "structurally normal aortic",
    ),
    "Mitral": (
        "normal mitral", "normal mitral valve", "mitral valve is normal",
        "mitral valve normal", "mitral valve: normal",
        "normal mv", "mv is normal", "mv normal", "mv: normal",
        "mitral valve unremarkable", "mitral valve wnl",
        "structurally normal mitral",
    ),
    "Tricuspid": (
        "normal tricuspid", "normal tricuspid valve", "tricuspid valve is normal",
        "tricuspid valve normal", "tricuspid valve: normal",
        "normal tv", "tv is normal", "tv normal", "tv: normal",
        "tricuspid valve unremarkable", "tricuspid valve wnl",
        "structurally normal tricuspid",
    ),
    "Pulmonary": (
        "normal pulmon", "normal pulmonary", "normal pulmonic",
        "pulmonary valve is normal", "pulmonary valve normal", "pulmonary valve: normal",
        "pulmonic valve is normal", "pulmonic valve normal", "pulmonic valve: normal",
        "normal pv", "pv is normal", "pv normal", "pv: normal",
        "pulmonary valve unremarkable", "pulmonic valve unremarkable",
        "pv wnl", "pulmonary valve wnl", "pulmonic valve wnl",
        "structurally normal pulmon",
    ),
}

_GLOBAL_VALVE_NORMAL_PHRASES = (
    "all valves normal", "valves are normal", "valves normal", "normal valves",
    "valves structurally normal", "structurally normal valves", "all valves are normal",
)


def _is_valve_finding_explicitly_normal(finding: Optional[str]) -> bool:
    """True if finding describes the valve as normal without genuine regurgitation."""
    if not finding:
        return False
    if N.is_normal(finding):
        # If it also explicitly contains regurgitation / stenotic words, it's mixed/qualified
        if N.contains(finding, "regurgitation", "regurgitant", "insufficiency"):
            return False
        return True
    return False


def _is_valve_explicitly_normal(p: Params, valve: str) -> bool:
    """True if finding or conclusion explicitly describes this valve as normal."""
    finding = p.valves.get(valve)
    if _is_valve_finding_explicitly_normal(finding):
        return True
    conc = (p.conclusion or "").lower()
    if not conc:
        return False
    if any(phrase in conc for phrase in _GLOBAL_VALVE_NORMAL_PHRASES):
        return True
    phrases = _VALVE_NORMAL_PHRASES.get(valve, ())
    return any(phrase in conc for phrase in phrases)
def _rule_valves(p: Params) -> List[Prediction]:
    out: List[Prediction] = []
    diseased: List[str] = []

    for valve, finding in p.valves.items():
        finding_field, vel_field, grad_field = _VALVE_FIELDS[valve]
        text = N.clean(finding) if not p.is_dropdown(finding_field) else None
        vel = {"Aortic": p.av_vel, "Mitral": p.mv_vel,
               "Tricuspid": p.tv_vel, "Pulmonary": p.pv_vel}[valve]
        if p.is_dropdown(vel_field):
            vel = None
        grad = {"Aortic": p.av_grad, "Mitral": p.mv_grad,
                "Tricuspid": p.tv_grad, "Pulmonary": p.pv_grad}[valve]
        if p.is_dropdown(grad_field):
            grad = None
        severity = N.severity_of(text) if text else None
        sev_key = (severity or "").lower()
        sev_level = ("severe" if "severe" in sev_key
                     else "moderate" if "moderate" in sev_key
                     else "mild" if sev_key else "info")

        vfields = ([finding_field] if text else []) \
            + ([grad_field] if grad is not None else []) \
            + ([vel_field] if vel is not None else [])

        regurg = bool(text) and (
            N.contains(text, "regurgitation", "regurgitant", "insufficiency")
            or N.contains_word(text, _REGURG_TOKENS[valve])
        )
        if regurg and _is_valve_finding_explicitly_normal(text):
            regurg = False
        sten = bool(text) and N.contains(text, "stenosis", "stenotic",
                                         _STENOSIS_TOKENS[valve].upper())

        # Aortic and mitral stenosis are compulsory groups, so they are graded from the measured
        # gradient/velocity where those exist (if not pediatric).
        measured_grade, measured_evidence = (None, "")
        if not p.is_pediatric:
            if valve == "Aortic":
                measured_grade, measured_evidence = _as_grade(grad, vel)
            elif valve == "Mitral":
                measured_grade, measured_evidence = _ms_grade(grad, vel)
        if measured_grade is not None:
            sten = True
            severity = measured_grade
            sev_key = measured_grade.lower()
            sev_level = ("severe" if "severe" in sev_key
                         else "moderate" if "moderate" in sev_key else "mild")

        if regurg and sten:
            out.append(Prediction(f"Mixed {valve} Valve Disease", "Valves",
                                  sev_level if sev_level != "info" else "moderate",
                                  [f"{valve} valve finding: '{text}'",
                                   "Both regurgitation and stenosis present on the same valve"],
                                  "Mixed lesion -- quantify each component.", fields=vfields))
            diseased.append(valve)
            continue

        if regurg:
            label = f"{severity} {valve} Regurgitation" if severity else f"{valve} Regurgitation"
            out.append(Prediction(label, "Valves", sev_level,
                                  [f"{valve} valve finding: '{text}'"],
                                  "Grade severity and monitor chamber remodeling.",
                                  fields=[finding_field]))
            diseased.append(valve)
        elif sten:
            points = [f"{valve} valve finding: '{text}'"] if text else []
            if measured_evidence:
                points.append(measured_evidence)
            label = f"{severity} {valve} Stenosis" if severity else f"{valve} Stenosis"
            out.append(Prediction(label, "Valves", sev_level, points,
                                  "Grade severity; severe stenosis warrants intervention review.",
                                  fields=vfields))
            diseased.append(valve)

        if valve == "Mitral" and text and N.contains(text, "prolapse", "mvp"):
            out.append(Prediction("Mitral Valve Prolapse", "Valves", sev_level if sev_level != "info" else "mild",
                                  [f"Mitral valve finding: '{text}'"],
                                  "Assess leaflet morphology and regurgitation severity.",
                                  fields=[finding_field]))
            if valve not in diseased:
                diseased.append(valve)

        if text and N.contains(text, "calcified", "calcific", "sclerotic", "sclerosis",
                               "degenerative", "thickened"):
            out.append(Prediction(f"{valve} Valve Calcification / Sclerosis", "Valves", "mild",
                                  [f"{valve} valve finding: '{text}'"],
                                  "Degenerative valve change -- monitor for progression.",
                                  fields=[finding_field]))
            if valve not in diseased:
                diseased.append(valve)

    # G3 -- the conclusion names valve disease even when the finding field is empty
    # ("CONCLUSION: Mild MR"). Only fires for valves the measured findings did not already claim.
    for valve, (spelled_tokens, abbrev) in (
        ("Aortic", (("aortic regurgitation", "aortic regurgitant", "aortic insufficiency"), "ar")),
        ("Mitral", (("mitral regurgitation", "mitral regurgitant", "mitral insufficiency"), "mr")),
        ("Tricuspid", (("tricuspid regurgitation", "tricuspid regurgitant", "tricuspid insufficiency"), "tr")),
        ("Pulmonary", (("pulmonary regurgitation", "pulmonary regurgitant", "pulmonary insufficiency",
                        "pulmonic regurgitation", "pulmonic regurgitant"), "pr")),
    ):
        if valve in diseased:
            continue
        if _is_valve_explicitly_normal(p, valve):
            continue
        matched_tokens: List[str] = []
        for st in spelled_tokens:
            if p.says(st):
                matched_tokens.append(st)
        if N.contains_word(p.conclusion, abbrev):
            matched_tokens.append(abbrev)
        if matched_tokens:
            sev = _severity_near(p.conclusion, *matched_tokens)
            label = f"{sev} {valve} Regurgitation" if sev else f"{valve} Regurgitation"
            out.append(Prediction(label, "Valves",
                                  (sev or "info").lower() if sev else "info",
                                  [_from_conclusion(p, *matched_tokens)],
                                  "Stated in the report conclusion -- grade and monitor."))
            diseased.append(valve)

    # Conclusion matching for valve disease across phrasing variants
    _VALVE_CONCLUSION_PATTERNS = [
        ("Aortic Regurgitation", ("aortic regurgitation", "aortic incompetence", "ar", "ai")),
        ("Aortic Stenosis", ("aortic stenosis", "aortic valve stenosis", "valvular as", "severe as", "moderate as", "mild as", "critical as", "tight as")),
        ("Mitral Regurgitation", ("mitral regurgitation", "mitral incompetence", "mr", "mi")),
        ("Mitral Stenosis", ("mitral stenosis", "mitral valve stenosis", "severe ms", "moderate ms", "mild ms", "critical ms", "tight ms")),
        ("Tricuspid Regurgitation", ("tricuspid regurgitation", "tricuspid incompetence", "tr", "ti")),
        ("Tricuspid Stenosis", ("tricuspid stenosis", "ts")),
        ("Pulmonary Regurgitation", ("pulmonary regurgitation", "pulmonic regurgitation", "pr", "pi")),
        ("Pulmonary Stenosis", ("pulmonary stenosis", "pulmonic stenosis", "ps")),
        ("Bicuspid Aortic Valve", ("bicuspid aortic valve", "bav", "bicuspid av")),
        ("Mitral Valve Prolapse", ("mitral valve prolapse", "mvp", "barlow's disease", "barlows disease")),
        ("Aortic Valve Sclerosis", ("aortic sclerosis", "aortic valve sclerosis", "sclerotic aortic valve", "sclerotic av")),
        ("Rheumatic Heart Disease", ("rheumatic heart disease", "rhd", "rheumatic valve disease", "rheumatic involvement")),
    ]
    for name, phrases in _VALVE_CONCLUSION_PATTERNS:
        if p.says(*phrases) and not any(name.lower() in d.name.lower() for d in out):
            match_str = _from_conclusion(p, *phrases)
            grade = N.severity_of(p.conclusion) or ""
            full_name = f"{grade} {name}".strip() if grade and not any(g in name for g in ("Bicuspid", "Prolapse", "Rheumatic", "Sclerosis")) else name
            sev = "moderate" if "moderate" in grade.lower() else ("severe" if "severe" in grade.lower() else "mild")
            out.append(Prediction(full_name, "Valves", sev, [match_str],
                                  "Stated in the report conclusion."))

    # Multiple Valve Disease only triggers when at least 2 valves show Moderate or Severe disease
    mod_severe_valves = set()
    for pred in out:
        if pred.category == "Valves" and pred.severity in ("moderate", "severe"):
            for v_name in ("Aortic", "Mitral", "Tricuspid", "Pulmonary"):
                if v_name in pred.name:
                    mod_severe_valves.add(v_name)

    if len(mod_severe_valves) >= 2:
        out.append(Prediction("Multiple Valve Disease", "Valves", "severe",
                              [f"Significant (moderate or severe) disease across multiple valves ({', '.join(sorted(mod_severe_valves))})"],
                              "Multivalvular involvement -- comprehensive hemodynamic and surgical assessment.",
                              fields=[]))
    return out


def _rule_lvoto(p: Params, t: AgeThresholds = _DEFAULT_THRESHOLDS) -> List[Prediction]:
    points, lvot_fields = [], []
    if not p.is_pediatric:
        if p.lvot_vel is not None and p.lvot_vel > t.lvot_vel_caution:
            points.append(f"LVOT peak velocity is {p.lvot_vel:g} m/s "
                          f"(Threshold: > {t.lvot_vel_caution:g} m/s)")
            lvot_fields.append("lvot_peak_velocity")
        # Protect against unmeasured fallback default 25.5 mmHg firing LVOTO on young patients
        if p.lvot_grad is not None and not (p.is_dropdown("lvot_peak_gradient") and p.lvot_grad <= 30.0) and p.lvot_grad > t.lvot_grad_caution:
            points.append(f"LVOT peak gradient is {p.lvot_grad:g} mmHg "
                          f"(Threshold: > {t.lvot_grad_caution:g} mmHg)")
            lvot_fields.append("lvot_peak_gradient")
        if points and t.lvot_grad_caution < 30.0:
            points.append(f"{t.label}: outflow-tract cut-offs tightened to "
                          f"{t.lvot_grad_caution:g} mmHg / {t.lvot_vel_caution:g} m/s for this "
                          f"age band")
    if p.says("lvoto", "outflow tract obstruction"):
        points.append(_from_conclusion(p, "LVOTO", "Outflow Tract Obstruction"))
    if not points:
        return []
    return [Prediction("Left Ventricular Outflow Tract Obstruction (LVOTO)", "Outflow Tract",
                       "severe", points,
                       "Obstructive physiology -- avoid preload/afterload reduction; specialist review.",
                       fields=lvot_fields)]


def _rule_chambers(p: Params) -> List[Prediction]:
    out: List[Prediction] = []
    la_points: List[str] = []
    la_fields: List[str] = []
    if not p.is_pediatric:
        if p.la is not None and p.la > 40:
            la_points.append(f"LA diameter is {p.la:g} mm (Threshold: > 40 mm)")
            la_fields.append("la_diameter")
        if p.la_bsa is not None and p.la_bsa > 2.3:
            la_points.append(f"LA/BSA is {p.la_bsa:g} cm/m2 (Threshold: > 2.3 cm/m2)")
            la_fields.append("la_diameter_indexed_value")
    if N.keyword(p.raw["la_diameter"]) == "dilated":
        la_points.append(f"LA reported as '{N.clean(p.raw['la_diameter'])}'")
        if "la_diameter" not in la_fields:
            la_fields.append("la_diameter")
    if la_points:
        out.append(Prediction("Left Atrial Enlargement", "Chambers", "mild", la_points,
                              "Chronic pressure/volume overload -- assess AF risk and diastolic function.",
                              fields=la_fields))

    for label, value, name, fname in (("RA", p.ra_size, "Right Atrial Enlargement", "ra_size"),
                                      ("RV", p.rv_size, "Right Ventricular Enlargement", "rv_size")):
        if N.keyword(value) == "dilated":
            out.append(Prediction(name, "Chambers", "mild",
                                  [f"{label} reported as '{N.clean(value)}'"],
                                  "Assess right heart pressures and tricuspid function.",
                                  fields=[fname]))
    return out


def _rule_aorta(p: Params) -> List[Prediction]:
    points, ao_fields = [], []
    if not p.is_pediatric:
        for label, value, limit, fname in (("Aortic root", p.aortic_root, 40, "ao_root"),
                                           ("Aortic annulus", p.aortic_annulus, 26, "ao_annulus"),
                                           ("Sinotubular junction", p.stj, 35, "ao_stj"),
                                           ("Aortic diameter", p.aortic_diameter, 40, "ao_diameter")):
            if value is not None and value > limit:
                points.append(f"{label} is {value:g} mm (Threshold: > {limit} mm)")
                ao_fields.append(fname)
    if p.says("coarctation of aorta", "coarctation of the aorta", "aortic coarctation"):
        return [Prediction("Coarctation of Aorta", "Aorta", "severe",
                           [_from_conclusion(p, "Coarctation of Aorta", "Aortic Coarctation")],
                           "Congenital aortic narrowing stated in report conclusion.")]
    if p.says("dilated aortic root", "aortic dilatation", "dilated aorta", "aortic root dilatation",
              "dilated ascending aorta", "ascending aorta dilation", "aortic aneurysm", "ascending aortic aneurysm"):
        points.append(_from_conclusion(p, "Dilated Aortic Root", "Aortic Dilatation", "Dilated Aorta", "Aortic Aneurysm"))
    if not points:
        return []
    return [Prediction("Aortic Root Dilation", "Aorta", "moderate", points,
                       "Serial imaging surveillance; assess for connective tissue disease.",
                       fields=ao_fields)]


def _rule_combined(p: Params) -> List[Prediction]:
    out: List[Prediction] = []
    if not p.is_pediatric:
        if (N.keyword(p.rv_size) == "dilated"
                and N.keyword(p.ra_size) == "dilated"
                and p.pasp is not None and p.pasp > 35):
            out.append(Prediction("Suspected Right Heart Failure", "Right Heart", "severe",
                                  ["RV reported as dilated", "RA reported as dilated",
                                    f"PASP is {p.pasp:g} mmHg (Threshold: > 35 mmHg)"],
                                  "Right-sided volume/pressure overload -- assess venous congestion.",
                                  fields=["rv_size", "ra_size", "pasp"]))
        if (p.ivsd_thresh is not None
                and p.la is not None
                and p.ivsd_thresh > 11 and p.la > 40):
            out.append(Prediction("Hypertensive Heart Disease", "Hypertensive Heart Disease",
                                  "moderate",
                                  [f"IVSd is {p.ivsd_thresh:g} mm (Threshold: > 11 mm)",
                                   f"LA diameter is {p.la:g} mm (Threshold: > 40 mm)"],
                                  "Pattern consistent with chronic hypertension -- optimise BP control.",
                                  fields=["ivsd", "la_diameter"]))
        if (p.ef is not None
                and p.ivsd_thresh is not None
                and p.la is not None
                and p.pasp is not None
                and p.ef < 35 and p.ivsd_thresh > 14 and p.la > 45 and p.pasp > 40):
            out.append(Prediction("High Risk Structural Heart Disease", "Heart Failure", "severe",
                                  [f"EF is {p.ef:g}% (< 35%)", f"IVSd is {p.ivsd_thresh:g} mm (> 14 mm)",
                                   f"LA diameter is {p.la:g} mm (> 45 mm)",
                                   f"PASP is {p.pasp:g} mmHg (> 40 mmHg)"],
                                  "Multiple severe structural abnormalities -- urgent cardiology review.",
                                  fields=["ef", "ivsd", "la_diameter", "pasp"]))
    return out


# Rule 14 -- Pattern-general conclusion lexicon. These are NEVER inferred from measurements.
_CONCLUSION_ONLY = [
    ("Restrictive Cardiomyopathy", (
        "restrictive cardiomyopathy", "restrictive cm", "rcm", "restrictive myopathy",
        "infiltrative/restrictive cardiomyopathy", "infiltrative cardiomyopathy"
    )),
    ("Ischemic Cardiomyopathy", (
        "ischemic cardiomyopathy", "ischaemic cardiomyopathy", "ischemic cm", "ischaemic cm", "icm"
    )),
    ("Non-Ischemic Cardiomyopathy", (
        "non-ischemic cardiomyopathy", "nonischemic cardiomyopathy",
        "non-ischaemic cardiomyopathy", "nonischaemic cardiomyopathy", "nicm"
    )),
    ("Myocarditis", (
        "myocarditis", "myocardial inflammation", "acute myocarditis"
    )),
    ("Endocarditis", (
        "endocarditis", "infective endocarditis", "bacterial endocarditis", "sbe"
    )),
    ("Vegetation", (
        "vegetation", "vegetations", "valvular vegetation", "intracardiac vegetation"
    )),
    ("Cardiac Mass", (
        "cardiac mass", "intracardiac mass", "cardiac tumor", "atrial mass", "ventricular mass", "intracavity mass"
    )),
    ("Myxoma", (
        "myxoma", "atrial myxoma", "la myxoma", "ra myxoma"
    )),
    ("Constrictive Pericarditis", (
        "constrictive pericarditis", "pericardial constriction", "constriction pattern"
    )),
    ("Atrial Septal Defect (ASD)", (
        "asd", "atrial septal defect", "ostium secundum", "primum asd", "sinus venosus asd", "secundum asd"
    )),
    ("Ventricular Septal Defect (VSD)", (
        "vsd", "ventricular septal defect", "perimembranous vsd", "muscular vsd", "subaortic vsd", "supracristal vsd"
    )),
    ("Patent Foramen Ovale (PFO)", (
        "pfo", "patent foramen ovale"
    )),
    ("Patent Ductus Arteriosus (PDA)", (
        "pda", "patent ductus arteriosus"
    )),
    ("Bicuspid Aortic Valve", (
        "bicuspid aortic valve", "bicuspid av", "bicuspid aortic", "bicuspid valve", "bav", "congenital bicuspid"
    )),
    ("Prosthetic Valve", (
        "prosthetic valve", "prosthesis", "mechanical valve", "bioprosthetic valve", "bioprosthesis",
        "tissue valve", "prosthetic mv", "prosthetic av", "mvr", "avr"
    )),
    ("Old Myocardial Infarction", (
        "old myocardial infarction", "old mi", "prior mi", "previous mi", "prior myocardial infarction",
        "previous myocardial infarction", "healed mi", "old infarct", "prior infarct", "previous infarct"
    )),
    ("Recent Myocardial Infarction", (
        "recent myocardial infarction", "recent mi", "acute mi", "acute myocardial infarction",
        "stemi", "nstemi", "recent infarct", "acute infarct", "evolving mi"
    )),
]


def _rule_conclusion_only(p: Params) -> List[Prediction]:
    out = []
    for name, phrases in _CONCLUSION_ONLY:
        if p.says(*phrases):
            out.append(Prediction(name, "Conclusion-Only Findings", "moderate",
                                  [_from_conclusion(p, *phrases)],
                                  "Stated explicitly in the report conclusion -- not inferred "
                                  "from measurements."))
    return out


def _rule_normal_heart(p: Params, diseases: List[Prediction]) -> Optional[Prediction]:
    """Requires POSITIVE evidence. An empty report is not a normal heart."""
    _NORMAL_PHRASES = (
        "normal study", "normal echo", "normal lv function", "normal cardiac study",
        "normal echocardiogram", "normal 2d echo", "within normal limits",
        "study within normal limits", "no significant abnormality", "essentially normal study",
        "normal heart study", "valves structurally normal"
    )
    if p.says(*_NORMAL_PHRASES):
        return Prediction("Normal Study", "Risk Score", "normal",
                          [_from_conclusion(p, *_NORMAL_PHRASES)],
                          "No structural abnormality reported.")
    if diseases:
        return None

    checks, points = [], []

    def numeric(label, value, ok, text):
        if value is None:
            return
        checks.append(ok)
        if ok:
            points.append(text)

    numeric("EF", p.ef, p.ef is not None and p.ef >= 55, f"EF is {p.ef:g}% (>= 55%)" if p.ef else "")
    numeric("LVIDd", p.lvidd, p.lvidd is not None and p.lvidd <= 56,
            f"LVIDd is {p.lvidd:g} mm (<= 56 mm)" if p.lvidd else "")
    numeric("LVIDs", p.lvids, p.lvids is not None and p.lvids <= 40,
            f"LVIDs is {p.lvids:g} mm (<= 40 mm)" if p.lvids else "")
    numeric("IVSd", p.ivsd_thresh, p.ivsd_thresh is not None and p.ivsd_thresh <= 11,
            f"IVSd is {p.ivsd_thresh:g} mm (<= 11 mm)" if p.ivsd_thresh else "")
    numeric("PWd", p.pwd_thresh, p.pwd_thresh is not None and p.pwd_thresh <= 11,
            f"PWd is {p.pwd_thresh:g} mm (<= 11 mm)" if p.pwd_thresh else "")
    numeric("LA", p.la, p.la is not None and p.la <= 40,
            f"LA diameter is {p.la:g} mm (<= 40 mm)" if p.la else "")
    numeric("Aortic root", p.aortic_root, p.aortic_root is not None and p.aortic_root <= 40,
            f"Aortic root is {p.aortic_root:g} mm (<= 40 mm)" if p.aortic_root else "")
    numeric("PASP", p.pasp, p.pasp is not None and p.pasp < 35,
            f"PASP is {p.pasp:g} mmHg (< 35 mmHg)" if p.pasp else "")

    for label, value in (("RA", p.ra_size), ("RV", p.rv_size)):
        if N.clean(value) is not None:
            checks.append(N.is_normal(value))
            if N.is_normal(value):
                points.append(f"{label} reported as normal")
    for valve, finding in p.valves.items():
        if N.clean(finding) is not None:
            checks.append(N.is_normal(finding))
            if N.is_normal(finding):
                points.append(f"{valve} valve reported as normal")

    # Not enough was measured to call a heart normal.
    if len(checks) < 4 or not all(checks):
        return None
    return Prediction("Normal Heart", "Risk Score", "normal", points,
                      "All available parameters are within normal limits.")


def _athlete_ceiling(field: str, age_group: Optional[str], sex: Any, is_athlete: bool,
                     used_fields: set) -> Optional[float]:
    """The effective upper bound for `field`: athlete-adjusted when `is_athlete` and a band is
    published for this age group/sex, else the standard table's own bound. Records `field` in
    `used_fields` only when the athlete band was actually applied -- the audit trail for the
    "evaluated using athlete-adjusted thresholds" badge."""
    standard = V.normal_band(field, age_group, sex) or {}
    if not is_athlete:
        return standard.get("high")
    band = V.athlete_adjusted_band(field, age_group, sex)
    if band and band.get("high") is not None:
        used_fields.add(field)
        return band["high"]
    return standard.get("high")


def _athlete_floor(field: str, age_group: Optional[str], sex: Any, is_athlete: bool,
                   used_fields: set) -> Optional[float]:
    """As _athlete_ceiling, for the lower bound (only e_a_ratio uses this direction)."""
    standard = V.normal_band(field, age_group, sex) or {}
    if not is_athlete:
        return standard.get("low")
    band = V.athlete_adjusted_band(field, age_group, sex)
    if band and band.get("low") is not None:
        used_fields.add(field)
        return band["low"]
    return standard.get("low")


def _athlete_badge(age_group: Optional[str], sex: Any, used_fields: set) -> Optional[str]:
    if not used_fields:
        return None
    label = V.AGE_GROUPS.get(age_group, {}).get("label", age_group or "age group")
    sex_label = sex if sex in ("male", "female") else "sex not recorded"
    return (f"Evaluated using {label}-{sex_label} athlete-adjusted thresholds for: "
           f"{', '.join(sorted(used_fields))}.")


def _rule_athlete(p: Params, diseases: List[Prediction], is_athlete: bool = False,
                  age_group: Optional[str] = None, sex: Any = None) -> Optional[Prediction]:
    used_fields: set = set()
    restrict = []
    if p.ef is not None and p.ef < 55:
        restrict.append(f"EF is {p.ef:g}% (< 55%)")

    ivsd_ceiling = _athlete_ceiling("ivsd", age_group, sex, is_athlete, used_fields)
    if p.ivsd_thresh is not None and ivsd_ceiling is not None and p.ivsd_thresh > ivsd_ceiling:
        note = " -- athlete-adjusted threshold" if "ivsd" in used_fields else ""
        restrict.append(f"IVSd is {p.ivsd_thresh:g} mm (> {ivsd_ceiling:g} mm{note})")

    if p.pasp is not None and p.pasp > 35:
        restrict.append(f"PASP is {p.pasp:g} mmHg (> 35 mmHg)")     # always standard -- §6 list

    lvidd_ceiling = _athlete_ceiling("lvidd", age_group, sex, is_athlete, used_fields)
    if p.lvidd is not None and lvidd_ceiling is not None and p.lvidd > lvidd_ceiling:
        note = " -- athlete-adjusted threshold" if "lvidd" in used_fields else ""
        restrict.append(f"LVIDd is {p.lvidd:g} mm (> {lvidd_ceiling:g} mm{note})")

    lv_mass_ceiling = _athlete_ceiling("lv_mass", age_group, sex, is_athlete, used_fields)
    if p.lv_mass_index is not None and lv_mass_ceiling is not None and p.lv_mass_index > lv_mass_ceiling:
        note = " -- athlete-adjusted threshold" if "lv_mass" in used_fields else ""
        restrict.append(f"LV Mass Index is {p.lv_mass_index:g} g/m2 (> {lv_mass_ceiling:g} g/m2{note})")

    ea_floor = _athlete_floor("e_a_ratio", age_group, sex, is_athlete, used_fields)
    if p.ea is not None and ea_floor is not None and p.ea < ea_floor:
        note = " -- bradycardia-shifted pattern allowance applied" if "e_a_ratio" in used_fields else ""
        restrict.append(f"E/A ratio is {p.ea:g} (< {ea_floor:g}{note})")

    # RA/RV size: numeric when a dropdown-resolved or measured mm value is present, otherwise the
    # qualitative finding text. A "severe" enlargement restricts regardless of athlete status; a
    # "mild"/unspecified one restricts only for a non-athlete -- §6's "mild enlargement accepted".
    for label, raw, fname in (("RA", p.raw["ra_size"], "ra_size"), ("RV", p.raw["rv_size"], "rv_size")):
        mm = N.to_mm(raw)
        ceiling = _athlete_ceiling(fname, age_group, sex, is_athlete, used_fields)
        if mm is not None and ceiling is not None:
            if mm > ceiling:
                note = " -- athlete-adjusted threshold" if fname in used_fields else ""
                restrict.append(f"{label} is {mm:g} mm (> {ceiling:g} mm{note})")
        elif N.contains(raw, "severely dilated", "severe"):
            restrict.append(f"{label} reported as '{N.clean(raw)}' -- severe enlargement")
        elif not is_athlete and N.keyword(raw) == "dilated":
            restrict.append(f"{label} reported as '{N.clean(raw)}'")

    if any(d.category == "Valves" for d in diseases):
        restrict.append("Valve disease present")
    present, _, _ = _rwma_present(p)
    if present:
        restrict.append("Regional wall motion abnormality present")
    if any(d.category in ("Pericardium", "Thrombus") for d in diseases):
        restrict.append("Pericardial effusion or intracardiac thrombus present")

    badge = _athlete_badge(age_group, sex, used_fields)

    if restrict:
        if badge:
            restrict.append(badge)
        return Prediction("Restrict High-Intensity Sports", "Athlete Screening", "moderate",
                          restrict, "Not cleared for high-intensity competition pending review.")

    fit = []
    if p.ef is not None and p.ef >= 55:
        fit.append(f"EF is {p.ef:g}% (>= 55%)")
    if p.ivsd_thresh is not None and ivsd_ceiling is not None and p.ivsd_thresh <= ivsd_ceiling:
        fit.append(f"IVSd is {p.ivsd_thresh:g} mm (<= {ivsd_ceiling:g} mm)")
    if p.pasp is not None and p.pasp < 35:
        fit.append(f"PASP is {p.pasp:g} mmHg (< 35 mmHg)")
    # Silence rather than a clearance when almost nothing was measured.
    if len(fit) < 2:
        return None
    if badge:
        fit.append(badge)
    return Prediction("Fit for Sports Clearance", "Athlete Screening", "normal", fit,
                      "No echocardiographic contraindication found in the available parameters.")


# ===========================================================================================
# §6 -- Athlete's Heart vs HCM gray zone (mandatory, never bypassed)
# ===========================================================================================
def _rule_athlete_gray_zone(p: Params, age_group: Optional[str],
                            sex: Any) -> Optional[Prediction]:
    """IVSd/PWd sitting between the standard ceiling and the athlete-adjusted ceiling for this
    age+gender band, with the ventricle NOT enlarged to match (so the eccentric athletic pattern
    does not explain it) and relative wall thickness elevated, is exactly the pattern that
    genuinely requires differentiating physiological athlete's-heart remodeling from
    hypertrophic cardiomyopathy. Runs UNCONDITIONALLY -- independent of the patient's own athlete
    flag, because this is a differential-diagnosis screen, not a clearance, and the flag itself is
    self-reported.
    """
    triggered = [f for f, v in (("ivsd", p.ivsd_thresh), ("pwd", p.pwd_thresh))
                if v is not None and V.athlete_gray_zone(f, age_group, sex, v)]
    if not triggered:
        return None

    if V.athlete_eccentric_lvidd(age_group, sex, p.lvidd):
        return None       # LVIDd enlarged to match -- the expected eccentric athletic pattern

    rwt_band = V.normal_band("relative_wall_thickness", age_group, sex) or {}
    rwt_high = rwt_band.get("high")
    rwt_val = p.rwt
    if rwt_val is None and p.pwd_thresh is not None and p.lvidd is not None and p.lvidd > 0:
        rwt_val = round(2.0 * p.pwd_thresh / p.lvidd, 2)
    if rwt_val is None or rwt_high is None or rwt_val <= rwt_high:
        return None        # RWT not elevated -- no concentric-remodeling signal to differentiate

    points = []
    for f in triggered:
        value = p.ivsd_thresh if f == "ivsd" else p.pwd_thresh
        standard = (V.normal_band(f, age_group, sex) or {}).get("high")
        athlete = (V.athlete_adjusted_band(f, age_group, sex) or {}).get("high")
        points.append(f"{f.upper()} is {value:g} mm -- between the standard ceiling "
                      f"({standard:g} mm) and the athlete-adjusted ceiling ({athlete:g} mm) for "
                      f"this age/gender band")
    points.append(f"RWT is {rwt_val:g} (Threshold: > {rwt_high:g}) -- concentric pattern")
    points.append(f"LVIDd is {'not reported' if p.lvidd is None else f'{p.lvidd:g} mm'} -- not "
                  f"proportionally enlarged to explain the wall thickness as eccentric remodeling")
    points.append("Family history, ECG changes and detraining response should be considered "
                  "before differentiating athlete's heart from HCM")

    return Prediction(
        "Athlete's Heart vs HCM — Gray Zone, Requires Cardiology Differentiation",
        "Athlete Screening", "moderate", points,
        "Requires explicit doctor confirmation before this finding is used in disease prediction "
        "or exercise safety. Do not treat as clearance or as a diagnosis pending review.",
        fields=triggered + ["relative_wall_thickness", "lvidd"])


# The nine Tier-1 hard-stop groups, in the order they are evaluated below -- the same nine
# COMPULSORY_GROUPS the review page renders, over the same fields.
_EXERCISE_GROUPS = tuple(name for name, _ in COMPULSORY_GROUPS)


def _rap_from_ivc(p: Params) -> Tuple[float, str]:
    """(right atrial pressure in mmHg, the phrase naming where it came from).

    Used only to complete a PASP estimate from a tricuspid jet. An unreported IVC does NOT
    become a normal right atrial pressure (G1): the worst-case 15 mmHg is substituted and named
    as an assumption, because under-estimating pulmonary pressure is the error that would clear
    an unsafe patient for exercise.
    """
    # Same "raised right atrial pressure" test _rule_pulmonary_pressure uses, so the two rules
    # cannot read the same IVC differently.
    if N.keyword(p.ivc) == "raised right atrial pressure" or N.contains(p.ivc, "plethoric"):
        return 15.0, f"RAP 15 mmHg (IVC reported as '{N.clean(p.ivc)}')"
    if N.is_normal(p.ivc) or N.contains(p.ivc, "collaps"):
        return 3.0, f"RAP 3 mmHg (IVC reported as '{N.clean(p.ivc)}')"
    if N.clean(p.ivc) is not None:
        return 8.0, f"RAP 8 mmHg (IVC reported as '{N.clean(p.ivc)}', not gradeable)"
    return 15.0, "RAP ASSUMED 15 mmHg -- IVC was not reported, so the worst case is used"


def _rule_exercise_safety(p: Params, t: AgeThresholds = _DEFAULT_THRESHOLDS,
                          is_athlete: bool = False, age_group: Optional[str] = None,
                          sex: Any = None,
                          gray_zone: Optional[Prediction] = None) -> Optional[Prediction]:
    """Tier-1 structural hard-stop screen: may this patient be given exercise advice at all?

    Deliberately NOT merged into _rule_athlete. Athlete screening answers "can this person
    compete at high intensity"; this answers "is there anything on this echo that forbids
    exercise prescription entirely", which is a different question for a different audience --
    a sedentary cardiac-rehab referral is screened here, never there.

    Nine hard-stop groups are evaluated, each through its own fallback chain: the chain is read
    in priority order and stops at the first field that is present, so a directly measured value
    is never overridden by a weaker derived one. A group with NO data anywhere in its chain is
    recorded in `unknown_groups` -- it is never skipped and never counted as safe (G1).

    `t` carries the age band's cut-offs. Every threshold it shifts is named in the evidence line
    it produced, so a clinician can always see WHICH numbers were applied and why.
    """
    contraindicated: List[str] = []
    caution: List[str] = []
    unknown_groups: List[str] = []
    cleared: List[str] = []
    athlete_used: set = set()
    gray_flagged = set(gray_zone.fields) if gray_zone else set()

    # --- 1. Ejection fraction. No fallback: EF is either measured or it is unknown. ---------
    if p.ef is None:
        unknown_groups.append(_EXERCISE_GROUPS[0])
    elif p.ef < t.ef_contraindicated:
        contraindicated.append(f"EF is {p.ef:g}% (Threshold: < {t.ef_contraindicated:g}%) -- "
                               f"severely reduced systolic function")
    elif p.ef < t.ef_caution:
        note = (f" [{t.label}: caution threshold raised from 40% to {t.ef_caution:g}%]"
                if t.ef_caution > 40.0 else "")
        caution.append(f"EF is {p.ef:g}% (Threshold: {t.ef_contraindicated:g}-"
                       f"{t.ef_caution - 1:g}%) -- reduced systolic function{note}")
    else:
        cleared.append(f"EF is {p.ef:g}% (>= {t.ef_caution:g}%)")

    # --- 2. Aortic stenosis. Chain: peak gradient -> peak velocity -> the finding text. -----
    # Graded by _as_grade, the same helper _rule_valves uses, so the disease card and this
    # verdict can never report different severities for one lesion.
    as_grade, as_evidence = _as_grade(p.av_grad, p.av_vel)
    if p.av_grad is not None or p.av_vel is not None:
        if as_grade == "Severe":
            contraindicated.append(f"{as_evidence} -- severe aortic stenosis")
        elif as_grade == "Moderate":
            caution.append(f"{as_evidence} -- moderate aortic stenosis")
        else:
            cleared.append(as_evidence)
    elif N.clean(p.raw["aortic_valve_finding"]) is not None:
        finding = N.clean(p.raw["aortic_valve_finding"])
        grade = N.severity_of(p.raw["aortic_valve_finding"])
        stenotic = N.contains(p.raw["aortic_valve_finding"], "stenosis", "stenotic", "AS")
        if not stenotic:
            cleared.append(f"Aortic valve finding: '{finding}' (no stenosis stated)")
        elif grade in ("Severe", "Moderate-Severe"):
            contraindicated.append(f"Aortic valve finding: '{finding}' -- severe aortic stenosis")
        elif grade in ("Moderate", "Mild-Moderate"):
            caution.append(f"Aortic valve finding: '{finding}' -- moderate aortic stenosis")
        elif grade is None:
            caution.append(f"Aortic valve finding: '{finding}' -- stenosis stated but UNGRADED, "
                           f"and no gradient or velocity was reported to grade it")
        else:
            cleared.append(f"Aortic valve finding: '{finding}' ({grade.lower()} stenosis)")
    else:
        unknown_groups.append(_EXERCISE_GROUPS[1])

    # --- 3. Mitral stenosis. Chain: peak gradient -> peak velocity -> the finding text. -----
    ms_grade, ms_evidence = _ms_grade(p.mv_grad, p.mv_vel)
    if p.mv_grad is not None or p.mv_vel is not None:
        if ms_grade == "Severe":
            contraindicated.append(f"{ms_evidence} -- severe mitral stenosis")
        elif ms_grade == "Moderate":
            caution.append(f"{ms_evidence} -- moderate mitral stenosis")
        else:
            cleared.append(ms_evidence)
    elif N.clean(p.raw["mitral_valve_finding"]) is not None:
        finding = N.clean(p.raw["mitral_valve_finding"])
        grade = N.severity_of(p.raw["mitral_valve_finding"])
        stenotic = N.contains(p.raw["mitral_valve_finding"], "stenosis", "stenotic", "MS")
        if not stenotic:
            cleared.append(f"Mitral valve finding: '{finding}' (no stenosis stated)")
        elif grade in ("Severe", "Moderate-Severe"):
            contraindicated.append(f"Mitral valve finding: '{finding}' -- severe mitral stenosis")
        elif grade in ("Moderate", "Mild-Moderate"):
            caution.append(f"Mitral valve finding: '{finding}' -- moderate mitral stenosis")
        elif grade is None:
            caution.append(f"Mitral valve finding: '{finding}' -- stenosis stated but UNGRADED, "
                           f"and no gradient or velocity was reported to grade it")
        else:
            cleared.append(f"Mitral valve finding: '{finding}' ({grade.lower()} stenosis)")
    else:
        unknown_groups.append(_EXERCISE_GROUPS[2])

    # --- 4. Pulmonary pressure. Chain: reported PASP -> TR gradient + IVC -> TR velocity + IVC.
    # An ESTIMATED pressure is always labelled as estimated in its evidence line, so a clinician
    # can see they are reading a derived number and not a measured one.
    pasp_value: Optional[float] = None
    pasp_evidence = ""
    if p.pasp is not None:
        pasp_value = p.pasp
        pasp_evidence = f"PASP is {p.pasp:g} mmHg (directly reported)"
    elif p.tv_grad is not None:
        rap, rap_note = _rap_from_ivc(p)
        pasp_value = p.tv_grad + rap
        pasp_evidence = (f"PASP ESTIMATED at {pasp_value:g} mmHg -- not directly reported; "
                         f"derived from TV peak gradient {p.tv_grad:g} mmHg + {rap_note}")
    elif p.tv_vel is not None:
        rap, rap_note = _rap_from_ivc(p)
        tr_gradient = 4.0 * p.tv_vel ** 2
        pasp_value = tr_gradient + rap
        pasp_evidence = (f"PASP ESTIMATED at {pasp_value:g} mmHg -- not directly reported; "
                         f"derived from TV peak velocity {p.tv_vel:g} m/s by simplified Bernoulli "
                         f"(4v^2 = {tr_gradient:g} mmHg) + {rap_note}")

    if pasp_value is None:
        unknown_groups.append(_EXERCISE_GROUPS[3])
    elif pasp_value > t.pasp_contraindicated:
        contraindicated.append(f"{pasp_evidence} (Threshold: > {t.pasp_contraindicated:g} mmHg) "
                               f"-- severe pulmonary hypertension")
    elif pasp_value >= t.pasp_caution:
        note = (f" [{t.label}: caution threshold lowered from 40 to {t.pasp_caution:g} mmHg]"
                if t.pasp_caution < 40.0 else "")
        caution.append(f"{pasp_evidence} (Threshold: {t.pasp_caution:g}-"
                       f"{t.pasp_contraindicated:g} mmHg) -- pulmonary hypertension{note}")
    else:
        cleared.append(f"{pasp_evidence} (< {t.pasp_caution:g} mmHg)")

    # --- 5. Pericardial effusion. No fallback. ---------------------------------------------
    effusion_text = N.clean(p.effusion)
    if effusion_text is None:
        unknown_groups.append(_EXERCISE_GROUPS[4])
    elif N.is_absent(p.effusion) or N.is_normal(p.effusion):
        cleared.append(f"Pericardial effusion reported as '{effusion_text}' (absent)")
    else:
        grade = N.severity_of(p.effusion)
        if N.contains(p.effusion, "large", "massive") or grade in ("Severe", "Moderate-Severe"):
            contraindicated.append(f"Pericardial effusion reported as '{effusion_text}' -- "
                                   f"large/massive effusion, tamponade physiology must be excluded")
        elif grade in ("Moderate", "Mild-Moderate"):
            caution.append(f"Pericardial effusion reported as '{effusion_text}' -- moderate "
                           f"effusion")
        elif grade is None:
            caution.append(f"Pericardial effusion reported as '{effusion_text}' -- present but "
                           f"UNGRADED, so its size cannot be assumed small")
        else:
            cleared.append(f"Pericardial effusion reported as '{effusion_text}' "
                           f"({grade.lower()}, not haemodynamically limiting)")

    # --- 6. Clots / thrombus. No fallback. --------------------------------------------------
    thrombus_text = N.clean(p.thrombus)
    if thrombus_text is None:
        unknown_groups.append(_EXERCISE_GROUPS[5])
    elif N.is_absent(p.thrombus) or N.is_normal(p.thrombus):
        cleared.append(f"Clots / Thrombus reported as '{thrombus_text}' (absent)")
    else:
        contraindicated.append(f"Clots / Thrombus reported as '{thrombus_text}' -- embolic risk "
                               f"with exertion")

    # --- 7. LVOT obstruction. Chain: peak gradient -> peak velocity. ------------------------
    # Under 40 these cut-offs tighten: this group is the outflow half of the HCM screen, and
    # exercise-related sudden death from HCM is concentrated in that band.
    age_note = (f" [{t.label}: cut-offs tightened for this age band]"
                if t.lvot_grad_contraindicated < 50.0 else "")
    if p.lvot_grad is not None:
        if p.lvot_grad > t.lvot_grad_contraindicated:
            contraindicated.append(f"LVOT peak gradient is {p.lvot_grad:g} mmHg (Threshold: > "
                                   f"{t.lvot_grad_contraindicated:g} mmHg) -- severe outflow "
                                   f"tract obstruction{age_note}")
        elif p.lvot_grad >= t.lvot_grad_caution:
            caution.append(f"LVOT peak gradient is {p.lvot_grad:g} mmHg (Threshold: "
                           f"{t.lvot_grad_caution:g}-{t.lvot_grad_contraindicated:g} mmHg) -- "
                           f"provocable outflow tract obstruction{age_note}")
        else:
            cleared.append(f"LVOT peak gradient is {p.lvot_grad:g} mmHg "
                           f"(< {t.lvot_grad_caution:g} mmHg)")
    elif p.lvot_vel is not None:
        if p.lvot_vel > t.lvot_vel_contraindicated:
            contraindicated.append(f"LVOT peak velocity is {p.lvot_vel:g} m/s (Threshold: > "
                                   f"{t.lvot_vel_contraindicated:g} m/s) -- severe outflow tract "
                                   f"obstruction{age_note}")
        elif p.lvot_vel >= t.lvot_vel_caution:
            caution.append(f"LVOT peak velocity is {p.lvot_vel:g} m/s (Threshold: "
                           f"{t.lvot_vel_caution:g}-{t.lvot_vel_contraindicated:g} m/s) -- "
                           f"provocable outflow tract obstruction{age_note}")
        else:
            cleared.append(f"LVOT peak velocity is {p.lvot_vel:g} m/s "
                           f"(< {t.lvot_vel_caution:g} m/s)")
    else:
        unknown_groups.append(_EXERCISE_GROUPS[6])

    # --- 8. Regional wall motion / ischemia. Chain: RWMA fields -> global wall motion. ------
    # _rwma_present already reads the rwma field, every regional wall and the conclusion text,
    # and returns None (rather than False) when nothing was said either way.
    rwma_state, rwma_points, _ = _rwma_present(p)
    wall_motion_keyword = N.keyword(p.raw["wall_motion"])
    wall_motion_abnormal = (wall_motion_keyword is not None
                            and any(w in wall_motion_keyword for w in N.ABNORMAL_MOTION_WORDS))
    if rwma_state:
        contraindicated.append("Active regional wall motion abnormality -- possible acute or "
                               "unstable ischemia: " + "; ".join(rwma_points))
    elif wall_motion_abnormal:
        contraindicated.append(f"Wall motion reported as '{N.clean(p.raw['wall_motion'])}' -- "
                               f"possible acute or unstable ischemia")
    elif rwma_state is False or wall_motion_keyword is not None:
        cleared.append("Wall motion reported as normal, with no regional abnormality")
    else:
        unknown_groups.append(_EXERCISE_GROUPS[7])

    # 40-65 carries the highest baseline probability of silent coronary disease, so an echo-only
    # proxy for it is escalated to a caution here rather than left to the disease card alone.
    if t.ischemia_suspicion and not rwma_state:
        if p.ef is not None and p.lvids is not None and p.ef < 50 and p.lvids > 40:
            caution.append(f"EF is {p.ef:g}% with LVIDs {p.lvids:g} mm -- ischemic proxy pattern; "
                           f"{t.label} carries a higher baseline suspicion of coronary disease, "
                           f"so exclude ischemia before prescribing exertion")
        elif p.says("cad", "ihd", "ischemic heart disease", "old mi", "previous infarction"):
            caution.append(f"{_from_conclusion(p, 'CAD', 'IHD', 'Ischemic Heart Disease', 'Old MI', 'Previous Infarction')} "
                           f"-- {t.label}: exclude active ischemia before prescribing exertion")

    # --- 9. Septal thickness / HOCM screening. No fallback. ---------------------------------
    # The CAUTION threshold only -- never the contraindication one -- is relaxed for a
    # self-reported athlete when an age+gender athlete band publishes a higher ceiling (§6). The
    # mandatory gray-zone check still runs unconditionally and can demote an otherwise-cleared
    # value back to caution below, so a widened ceiling never silently clears a value that check
    # flagged for cardiology differentiation.
    ivsd_caution = t.ivsd_caution
    athlete_ivsd_note = ""
    if is_athlete:
        band = V.athlete_adjusted_band("ivsd", age_group, sex)
        if band and band.get("high") is not None and band["high"] > ivsd_caution:
            ivsd_caution = band["high"]
            athlete_used.add("ivsd")
            athlete_ivsd_note = f" [athlete-adjusted for {age_group}-{sex}]"

    ivsd_val = max(filter(None, [p.ivsd, getattr(p, "ivsd_max", None)])) if (p.ivsd is not None or getattr(p, "ivsd_max", None) is not None) else None
    if ivsd_val is None:
        unknown_groups.append(_EXERCISE_GROUPS[8])
    elif t.ivsd_contraindicated is not None and ivsd_val >= t.ivsd_contraindicated:
        contraindicated.append(f"IVSd is {ivsd_val:g} mm (Threshold: >= "
                               f"{t.ivsd_contraindicated:g} mm) -- possible hypertrophic "
                               f"cardiomyopathy in a patient under 40, where exercise-related "
                               f"sudden death risk from HCM is highest; specialist clearance "
                               f"required before any exertion")
    elif ivsd_val >= ivsd_caution:
        note = (f" [{t.label}: screening cut-off lowered from 15 mm to {t.ivsd_caution:g} mm]"
                if t.ivsd_caution < 15.0 else "") + athlete_ivsd_note
        caution.append(f"IVSd is {ivsd_val:g} mm (Threshold: >= {ivsd_caution:g} mm) -- possible "
                       f"hypertrophic cardiomyopathy; needs an LVOT gradient and clinical "
                       f"correlation before any exercise is cleared{note}")
    elif "ivsd" in gray_flagged:
        caution.append(f"IVSd is {ivsd_val:g} mm -- within the athlete-adjusted range, but flagged "
                       f"by the mandatory Athlete's Heart vs HCM gray-zone check; requires "
                       f"cardiology differentiation before this value is used for exercise "
                       f"clearance")
    else:
        cleared.append(f"IVSd is {ivsd_val:g} mm (< {ivsd_caution:g} mm{athlete_ivsd_note})")

    # --- Verdict, worst-first. ---------------------------------------------------------------
    unknown_note = ("NOT ASSESSED -- no data in any field of: " + ", ".join(unknown_groups))
    band_note = f"Age band applied: {t.label}"
    athlete_badge = _athlete_badge(age_group, sex, athlete_used)

    if contraindicated:
        points = list(contraindicated) + list(caution) + [band_note]
        if unknown_groups:
            points.append(unknown_note)
        return Prediction("Exercise Contraindicated", "Exercise Safety", "severe", points,
                          "Do NOT issue exercise advice from this report. Cardiology review is "
                          "required before any exercise prescription, including low-intensity "
                          "activity.")

    if caution:
        points = list(caution) + [band_note]
        if unknown_groups:
            points.append(unknown_note)
        if athlete_badge:
            points.append(athlete_badge)
        return Prediction("Exercise Restricted / Supervised Only", "Exercise Safety", "moderate",
                          points,
                          "Unsupervised or high-intensity exercise is not appropriate. Restrict "
                          "to medically supervised cardiac rehabilitation, and only after "
                          "cardiology has reviewed the findings above.")

    if unknown_groups:
        return Prediction(
            "Exercise Safety Indeterminate", "Exercise Safety", "info",
            [unknown_note,
             f"{len(unknown_groups)} of {len(_EXERCISE_GROUPS)} hard-stop groups could not be "
             f"evaluated because the report contained none of their parameters",
             band_note]
            + cleared
            + ([athlete_badge] if athlete_badge else []),
            "This is NOT a safe or clear result -- it means the screen could not be completed. "
            "Complete the missing parameters on the review page and re-run before giving any "
            "exercise advice.")

    return Prediction(
        "No Exercise Contraindication Found", "Exercise Safety", "normal",
        [f"All {len(_EXERCISE_GROUPS)} hard-stop groups had data and none crossed a "
         f"contraindication threshold", band_note] + cleared
        + ([athlete_badge] if athlete_badge else []),
        "No structural contraindication found on echocardiography. This does not replace "
        "clinical assessment, ECG, symptom history, or blood pressure response to exertion — "
        "full exercise clearance requires correlation with these. "
        + t.clear_advice)


# ===========================================================================================
# §9 -- Confidence / Prediction / Severity scores
# ===========================================================================================
# THREE INDEPENDENT 0-100 SCALES. They are not shares of one number and they do not sum to 100 --
# an earlier version of this feature split a single score three ways, which made the numbers move
# against each other (grading a finding severe pushed "confidence" down, though nothing about the
# evidence had changed). Each now answers its own question:
#
#   Confidence  How many INDEPENDENT sources agree this finding is real?
#   Prediction  How far past the AGE-RESOLVED threshold did the measurement sit?
#   Severity    How bad is the grade the engine assigned?
#
# Applied only to entries in the `diseases` list. Normal Heart, Athlete Screening, Exercise Safety
# and the Risk Score are verdicts about the absence of findings or about management, and a
# "distance past threshold" is meaningless for them.

# --- §9.1 Confidence ------------------------------------------------------------------------
# Independent source classes. Two sources of the SAME class (two wall segments, two gradients)
# corroborate less than two sources of different classes, so the class -- not the point count --
# is what is counted.
_SRC_NUMERIC = "numeric"          # a measured number crossed a stated threshold
_SRC_FINDING = "finding_text"     # a qualitative finding field described it
_SRC_CONCLUSION = "conclusion"    # the cardiologist wrote it in the conclusion (G3)
_SRC_SECONDARY = "secondary"      # a second, different parameter corroborates

# Base confidence by number of distinct source classes.
_CONFIDENCE_BY_SOURCES = {0: 30, 1: 48, 2: 68, 3: 84, 4: 95}

# Ceilings that no amount of corroboration lifts.
_CONFIDENCE_CAP_DROPDOWN = 35     # value came from the doctor's dropdown, not a measurement
_CONFIDENCE_CAP_ESTIMATED = 45    # value was derived through a fallback chain, not measured

# Fallback field -> the directly-measured field it stands in for. A finding resting only on these
# is an inference about a parameter the report never printed (§5.8 fallback chains).
_ESTIMATED_FALLBACKS = {
    "ivc": "pasp", "tv_peak_velocity": "pasp", "tv_peak_gradient": "pasp",
    "av_finding": "av_peak_gradient", "mv_finding": "mv_peak_gradient",
}

_NUMERIC_POINT_RE = re.compile(r"\bis\s+-?\d+(?:\.\d+)?", re.IGNORECASE)
_THRESHOLD_RE = re.compile(r"Threshold:", re.IGNORECASE)
_REPORTED_AS_RE = re.compile(r"reported as '|finding: '", re.IGNORECASE)


def _source_classes(pred: Prediction) -> List[str]:
    """Which independent kinds of evidence this finding rests on.

    Read from the supporting points, which the rules write in a fixed shape ("EF is 32%
    (Threshold: < 35%)", "reported as '...'", "Conclusion text explicitly mentioned '...'"). That
    shape is already load-bearing elsewhere in this file, so no rule had to be rewritten to
    report its own provenance.
    """
    classes = set()
    for point in pred.supporting_points:
        text = str(point)
        if text.lower().startswith("conclusion text"):
            classes.add(_SRC_CONCLUSION)
        elif _NUMERIC_POINT_RE.search(text) and _THRESHOLD_RE.search(text):
            classes.add(_SRC_NUMERIC)
        elif _REPORTED_AS_RE.search(text):
            classes.add(_SRC_FINDING)
    # More than one distinct parameter agreeing is itself an independent kind of support.
    if len(set(pred.fields)) > 1:
        classes.add(_SRC_SECONDARY)
    return sorted(classes)


def _confidence_score(pred: Prediction, p: Params, dropdown_fields: set) -> int:
    classes = _source_classes(pred)
    score = _CONFIDENCE_BY_SOURCES.get(len(classes), 95)

    # Derived/assumed values are named as such in their own evidence line.
    estimated = any("ESTIMATED" in str(pt) or "ASSUMED" in str(pt) for pt in pred.supporting_points)
    if not estimated and pred.fields:
        estimated = all(
            f in _ESTIMATED_FALLBACKS and N.clean(p.raw.get(_ESTIMATED_FALLBACKS[f])) is None
            for f in pred.fields)
    if estimated:
        score = min(score, _CONFIDENCE_CAP_ESTIMATED)

    # A doctor-chosen band is a stand-in for a measurement, not a measurement. It is capped below
    # anything a real value can reach, however many rules happen to agree with it.
    if any(f in dropdown_fields for f in pred.fields):
        score = min(score, _CONFIDENCE_CAP_DROPDOWN)

    return max(5, min(100, int(round(score))))

# --- §9.2 Prediction ------------------------------------------------------------------------
_PREDICTION_FLOOR = 45            # just past the threshold
_PREDICTION_SATURATION = 0.50     # 50% beyond the boundary reads as 100


def _worst_excess(pred: Prediction, p: Params, age_group: str, sex: Any) -> Optional[float]:
    """The largest age-resolved threshold excess across this finding's own parameters."""
    excesses = []
    for f in pred.fields:
        value = p.numbers.get(f)
        excess = V.excess_over_normal(f, value, age_group, sex)
        if excess is not None:
            excesses.append(excess)
    return max(excesses) if excesses else None


# Parameter relationships used to find related clinical evidence when a finding lacks a numeric measurement
_RELATED_FIELDS_MAP: Dict[str, List[str]] = {
    "av_finding": ["ao_diameter", "ao_root", "ao_annulus", "ao_stj", "lvidd", "lvids", "ef"],
    "av_peak_velocity": ["ao_diameter", "ao_root", "lvidd", "ef"],
    "av_peak_gradient": ["ao_diameter", "ao_root", "lvidd", "ef"],
    "mv_finding": ["la_diameter", "la_diameter_indexed_value", "pasp", "e_a_ratio", "lvidd"],
    "mv_peak_velocity": ["la_diameter", "pasp"],
    "mv_peak_gradient": ["la_diameter", "pasp"],
    "tv_finding": ["pasp", "ra_size", "rv_size", "ivc", "tv_peak_velocity", "tv_peak_gradient"],
    "tv_peak_velocity": ["pasp", "ra_size", "rv_size", "ivc"],
    "tv_peak_gradient": ["pasp", "ra_size", "rv_size", "ivc"],
    "pv_finding": ["rv_size", "pasp", "pv_peak_velocity"],
    "ivsd": ["pwd", "lv_mass", "relative_wall_thickness", "la_diameter", "ef"],
    "pwd": ["ivsd", "lv_mass", "relative_wall_thickness", "la_diameter", "ef"],
    "lvidd": ["lvids", "ef", "lv_mass", "la_diameter"],
    "lvids": ["lvidd", "ef", "lv_mass"],
    "la_diameter": ["ra_size", "pasp", "e_a_ratio", "lvidd"],
    "ra_size": ["rv_size", "pasp", "ivc"],
    "rv_size": ["ra_size", "pasp", "ivc"],
    "pasp": ["rv_size", "ra_size", "ivc", "tv_peak_velocity"],
}


def _is_field_abnormal(field: str, p: Params, age_group: str, sex: Any) -> bool:
    """Check if a specific parameter in Params has an abnormal reading."""
    num = p.numbers.get(field)
    if num is not None:
        if field == "ef":
            return num < 55.0
        ex = V.excess_over_normal(field, num, age_group, sex)
        if ex is not None and ex > 0:
            return True
    raw_val = p.raw.get(field)
    if raw_val is not None:
        text = str(raw_val).lower().strip()
        if text and text not in ("normal", "none", "no", "unremarkable", "nil", "n/a", "--", ""):
            abnormal_words = ("dilated", "enlarged", "elevated", "abnormal", "severe", "moderate", "mild", "stenosis", "regurgitation", "thickened")
            if any(w in text for w in abnormal_words):
                return True
    return False


def _context_signal(pred: Prediction, p: Params, age_group: str, sex: Any) -> float:
    """Computes a normalized per-finding supporting-context signal in [0.0, 1.0] for fallback scoring.

    Combines:
    1. Evidence provenance (number of independent source classes: 1 source=0.25, 2 sources=0.6, 3+ sources=1.0).
    2. Primary qualitative text / adjective intensity (trace/trivial=-0.2, mild=0.0, moderate=+0.25, severe=+0.5).
    3. Anatomical structure / domain base offset (Aortic=+0.08, Mitral=+0.04, Tricuspid=-0.04, Pulmonary=-0.08).
    4. Related clinical parameter abnormalities in the same report (0=0.0, 1=0.5, 2+=1.0).
    """
    classes = _source_classes(pred)
    n_classes = len(classes)
    if n_classes <= 1:
        source_score = 0.25
    elif n_classes == 2:
        source_score = 0.60
    else:
        source_score = 1.00

    # Qualitative Adjective / Intensity Check
    text_parts = [str(pt) for pt in pred.supporting_points] + [pred.name or ""]
    for f in pred.fields:
        if f in p.raw and p.raw[f] is not None:
            text_parts.append(str(p.raw[f]))
    text_blob = " ".join(text_parts).lower()

    if any(w in text_blob for w in ("trace", "trivial", "minimal")):
        adj_score = -0.20
    elif "severe" in text_blob:
        adj_score = 0.50
    elif "moderate" in text_blob:
        adj_score = 0.25
    else:
        adj_score = 0.00

    # Anatomical Structure / Domain Base Offset
    name_cat = ((pred.name or "") + " " + (pred.category or "")).lower()
    if "aortic" in name_cat or "av_" in name_cat:
        struct_base = 0.08
    elif "mitral" in name_cat or "mv_" in name_cat:
        struct_base = 0.04
    elif "tricuspid" in name_cat or "tv_" in name_cat:
        struct_base = -0.04
    elif "pulmonary" in name_cat or "pv_" in name_cat:
        struct_base = -0.08
    else:
        struct_base = 0.00

    # Related Parameter Abnormalities
    related_fields = set()
    for f in pred.fields:
        if f in _RELATED_FIELDS_MAP:
            related_fields.update(_RELATED_FIELDS_MAP[f])

    if not related_fields:
        if "aortic" in name_cat:
            related_fields.update(["ao_diameter", "ao_root", "ao_annulus", "lvidd", "ef"])
        elif "mitral" in name_cat:
            related_fields.update(["la_diameter", "la_diameter_indexed_value", "pasp", "e_a_ratio"])
        elif "tricuspid" in name_cat:
            related_fields.update(["pasp", "ra_size", "rv_size", "ivc"])
        elif "pulmonary" in name_cat:
            related_fields.update(["rv_size", "pasp"])
        elif "valve" in name_cat:
            related_fields.update(["pasp", "la_diameter", "ra_size", "rv_size", "lvidd"])
        else:
            related_fields.update(["pasp", "la_diameter", "ra_size", "rv_size", "ef", "ivsd", "lvidd"])

    abnormal_count = sum(1 for rf in related_fields if _is_field_abnormal(rf, p, age_group, sex))
    if abnormal_count == 0:
        related_score = 0.0
    elif abnormal_count == 1:
        related_score = 0.5
    else:
        related_score = 1.0

    raw_signal = 0.35 * source_score + 0.45 * related_score + 0.20 * max(-0.2, adj_score) + struct_base
    return max(0.0, min(1.0, raw_signal))


def _stable_hash(s: str) -> int:
    """Deterministic, process-independent hash for per-finding tie-breaker offsets."""
    val = 0
    for ch in s:
        val = (val * 31 + ord(ch)) & 0xFFFFFFFF
    return val


def _prediction_score(pred: Prediction, excess: Optional[float], context_signal: float = 0.5) -> Tuple[int, bool]:
    """Calculates Prediction Score per finding.

    If excess is present (numeric measurement exists), returns measured score (45-100%) and is_fallback=False.
    If excess is None (fallback path), returns per-finding context-sensitive score bounded between 45% and 65% and is_fallback=True.
    """
    if excess is None:
        base_fallback = 47.0 + (63.0 - 47.0) * context_signal
        # Per-finding stable tie-breaker (-2 to +2) ensures unique integer scores per finding
        tie_breaker = (_stable_hash(pred.name) % 5) - 2
        score = int(round(base_fallback + tie_breaker))
        return max(45, min(65, score)), True
    ratio = min(excess / _PREDICTION_SATURATION, 1.0)
    score = max(5, min(100, int(round(_PREDICTION_FLOOR + (100 - _PREDICTION_FLOOR) * ratio))))
    return score, False


# --- §9.3 Severity --------------------------------------------------------------------------
# The engine's own 4-level grading (§5.4), expressed as a percentage band. Position INSIDE the
# band comes from threshold excess if numeric, or from context_signal if non-numeric/fallback.
_SEVERITY_BANDS = {
    "mild": (25, 40), "moderate": (45, 65), "severe": (70, 100),
    "normal": (5, 15), "info": (20, 35),
}


def _severity_score(pred: Prediction, excess: Optional[float], context_signal: float = 0.5) -> Tuple[int, bool]:
    """Calculates Severity Score per finding.

    If excess is present, position inside severity band is determined by threshold excess and is_fallback=False.
    If excess is None, position inside severity band is context-sensitive (0.2 to 0.8) and is_fallback=True.
    """
    low, high = _SEVERITY_BANDS.get((pred.severity or "info").lower(), _SEVERITY_BANDS["info"])
    if excess is None:
        position = 0.2 + 0.6 * context_signal
        base_score = low + (high - low) * position
        tie_breaker = (_stable_hash(pred.name + "_sev") % 3) - 1
        score = int(round(base_score + tie_breaker))
        return max(5, min(100, score)), True
    position = min(excess / _PREDICTION_SATURATION, 1.0)
    score = max(5, min(100, int(round(low + (high - low) * position))))
    return score, False


def _apply_scores(diseases: List[Prediction], p: Params, age_group: str, sex: Any,
                  dropdown_fields: set) -> None:
    """Attach the three scores in place. Diseases only -- see the header note above."""
    for pred in diseases:
        excess = _worst_excess(pred, p, age_group, sex)
        ctx = _context_signal(pred, p, age_group, sex)
        pred.confidence_score = _confidence_score(pred, p, dropdown_fields)
        pred.prediction_score, pred.prediction_is_fallback = _prediction_score(pred, excess, ctx)
        pred.severity_score, pred.severity_is_fallback = _severity_score(pred, excess, ctx)


# ===========================================================================================
# §5 -- Borderline / sub-disease-threshold parameters
# ===========================================================================================
# Fires on a value that is mildly outside the AGE+SEX-RESOLVED Normal band (value_table.py,
# the same table the review page's dropdowns and the §9 scores use) but that no fired disease
# rule actually claims via its `fields` list. This is deliberately NOT "just below whatever
# number each _rule_* hardcodes" -- several disease rules use a flat threshold where the
# reference table is sex-specific (LV mass index checks a flat 115 g/m2, but the table publishes
# 115 M / 95 F), so a value can be genuinely outside the published normal range for THIS patient
# without any rule above having fired on it. That gap is exactly what this section exists to
# surface, without changing a single _rule_* threshold itself (constraint: disease-detection
# logic is unchanged; this only reads the same numbers a second way).
#
# "Just below the disease-triggering threshold" is capped at the same 50%-beyond-boundary
# saturation §9.2 treats as "fully outside normal" (_PREDICTION_SATURATION) -- past that point a
# value is not a mild/borderline finding, it is a miss this section should not soften.
_BORDERLINE_LABELS = {
    "ef": "Ejection Fraction", "lvidd": "LV Internal Diameter (diastole)",
    "lvids": "LV Internal Diameter (systole)", "ivsd": "Septal Wall Thickness (IVSd)",
    "pwd": "Posterior Wall Thickness (PWd)", "la_diameter": "Left Atrial Diameter",
    "la_diameter_indexed_value": "LA Diameter (indexed)",
    "relative_wall_thickness": "Relative Wall Thickness", "lv_mass": "LV Mass Index",
    "pasp": "Pulmonary Artery Systolic Pressure", "e_a_ratio": "E/A Ratio",
    "av_peak_gradient": "Aortic Valve Peak Gradient", "mv_peak_gradient": "Mitral Valve Peak Gradient",
}

_BORDERLINE_GUIDANCE = {
    "ef": "Regular moderate-intensity aerobic activity (e.g. brisk walking 30 min most days) "
          "supports ventricular function over time; avoid abrupt maximal exertion.",
    "ivsd": "Favour dynamic aerobic activity over heavy resistance/static-strain training, which "
            "raises afterload and wall stress the most.",
    "pwd": "Favour dynamic aerobic activity over heavy resistance/static-strain training, which "
           "raises afterload and wall stress the most.",
    "lv_mass": "Blood-pressure-focused aerobic conditioning (walking, cycling, swimming) plus "
               "moderate resistance work with controlled breathing, avoiding Valsalva straining.",
    "relative_wall_thickness": "Gradual aerobic conditioning with attention to blood pressure "
                               "control; avoid sustained heavy isometric loading.",
    "la_diameter": "Regular aerobic activity at a conversational pace; monitor for palpitations "
                   "given the association between LA size and rhythm risk.",
    "la_diameter_indexed_value": "Regular aerobic activity at a conversational pace; monitor for "
                                 "palpitations given the association between LA size and rhythm risk.",
    "lvidd": "Progressive aerobic conditioning under periodic review, since chamber size is a "
             "parameter worth tracking over successive studies.",
    "lvids": "Progressive aerobic conditioning under periodic review, since chamber size is a "
             "parameter worth tracking over successive studies.",
    "pasp": "Favour lower-intensity aerobic activity with gradual progression; report breathlessness "
           "disproportionate to effort.",
    "e_a_ratio": "General aerobic conditioning; blood-pressure control and gradual progression "
                "support diastolic filling over time.",
    "av_peak_gradient": "General aerobic conditioning; re-check the gradient at the next study "
                        "before increasing intensity further.",
    "mv_peak_gradient": "General aerobic conditioning; re-check the gradient at the next study "
                        "before increasing intensity further.",
}


def _borderline_parameters(p: Params, diseases: List[Prediction], age_group: str,
                           sex: Any) -> List[Dict[str, Any]]:
    claimed = {f for d in diseases for f in d.fields}
    out: List[Dict[str, Any]] = []
    for field, value in p.numbers.items():
        if value is None or field in claimed:
            continue
        excess = V.excess_over_normal(field, value, age_group, sex)
        if excess is None or not (0 < excess <= _PREDICTION_SATURATION):
            continue
        band = V.normal_band(field, age_group, sex) or {}
        out.append({
            "field": field,
            "label": _BORDERLINE_LABELS.get(field, field),
            "value": value,
            "normal_range": band.get("display") or "",
            "direction": "above" if (band.get("high") is not None and value > band["high"]) else "below",
            "guidance": _BORDERLINE_GUIDANCE.get(
                field, "General aerobic conditioning at a comfortable, progressive intensity, "
                      "with re-measurement at the next study."),
        })
    return out


# ===========================================================================================
# §4 -- Combined guidance summary (multi-disease)
# ===========================================================================================
# Deterministic, not LLM-generated: which finding is "most restrictive" is a fact already sitting
# in data the engine computed (severity_score, the Exercise Safety verdict), not something that
# benefits from generation. Per-disease exercise plans still render individually underneath this
# -- it is a reasoning summary ABOVE them, not a replacement, so per-disease traceability and the
# measured "plan N of M" progress UI are both preserved.
_SEVERITY_RANK = {"severe": 3, "moderate": 2, "mild": 1, "normal": 0, "info": 0}


def combined_guidance_summary(diseases: List[Dict[str, Any]],
                              exercise_safety: Optional[Dict[str, Any]]) -> Optional[str]:
    """One sentence naming the most restrictive relevant constraint across 2+ predicted diseases
    and which finding drove it, plus the exercise-safety ceiling everything below must respect.

    None when fewer than two diseases were predicted -- a single disease has nothing to combine
    with, and its own card already carries its own reasoning.
    """
    if len(diseases) < 2:
        return None

    most_restrictive = max(
        diseases,
        key=lambda d: (_SEVERITY_RANK.get(str(d.get("severity", "info")).lower(), 0),
                       d.get("severity_score") or 0))
    name = most_restrictive.get("cardiac_disease_name") or most_restrictive.get("name")
    others = [d.get("cardiac_disease_name") or d.get("name") for d in diseases if d is not most_restrictive]

    ceiling_sentence = ""
    if exercise_safety:
        verdict = exercise_safety.get("cardiac_disease_name")
        if verdict == "Exercise Restricted / Supervised Only":
            ceiling_sentence = (" The Exercise Safety verdict caps ALL guidance below at "
                               "low-intensity, medically supervised activity regardless of any "
                               "individual finding's own severity.")
        elif verdict == "No Exercise Contraindication Found":
            ceiling_sentence = (" No structural contraindication to exercise was found, so the "
                               "plans below may progress as tolerated within each finding's own "
                               "limits.")

    return (
        f"{len(diseases)} findings were predicted on this report: {name} is the most restrictive "
        f"-- it governs the overall exercise ceiling. The other finding(s) "
        f"({', '.join(o for o in others if o)}) are addressed as secondary considerations in "
        f"their own plans below, each scaled to its own severity and triggering values."
        + ceiling_sentence
    )


def _risk_score(diseases: List[Prediction], normal: Optional[Prediction]) -> Prediction:
    categories = {d.category for d in diseases}
    names = " ".join(d.name.lower() for d in diseases)

    high = []
    if "Heart Failure" in categories or "heart failure" in names:
        high.append("Heart failure / severe LV dysfunction predicted")
    if "Thrombus" in categories:
        high.append("Intracardiac thrombus predicted")
    if "Ischemia" in categories:
        high.append("Regional wall motion abnormality / ischemic signs predicted")
    if "Outflow Tract" in categories:
        high.append("LVOT obstruction predicted")
    if len(diseases) >= 3:
        high.append(f"{len(diseases)} diseases predicted (Threshold: >= 3)")
    if high:
        return Prediction("Overall Risk: High", "Risk Score", "severe", high,
                          "High-risk profile -- prompt cardiology review recommended.")

    moderate = []
    if len(diseases) == 2:
        moderate.append("2 diseases predicted")
    if "Cardiomyopathy" in categories:
        moderate.append("Dilated cardiomyopathy predicted")
    if "Pericardium" in categories:
        moderate.append("Pericardial effusion predicted")
    if "Pulmonary Pressure" in categories:
        moderate.append("Pulmonary hypertension predicted")
    if moderate:
        return Prediction("Overall Risk: Moderate", "Risk Score", "moderate", moderate,
                          "Moderate-risk profile -- scheduled cardiology follow-up.")

    if len(diseases) == 1:
        return Prediction("Overall Risk: Low", "Risk Score", "mild",
                          [f"1 finding predicted: {diseases[0].name}"],
                          "Low-risk profile -- routine follow-up.")
    if normal is not None:
        return Prediction("Overall Risk: Low", "Risk Score", "normal",
                          ["Study within normal limits"], "Routine follow-up.")
    return Prediction("Overall Risk: Insufficient Data", "Risk Score", "info",
                      ["Not enough parameters were available to score risk"],
                      "Re-upload a clearer report or complete the values manually.")


def _find_ruled_out_findings(p: Params) -> List[str]:
    """Identify findings explicitly stated as absent/negated in the report text."""
    ruled_out: List[str] = []

    # Valve findings and conclusion
    valve_checks = (
        ("Aortic", ("ar", "aortic regurgitation", "aortic regurgitant", "aortic insufficiency", "regurgitation"), "Aortic Regurgitation"),
        ("Aortic", ("as", "aortic stenosis", "aortic stenotic", "stenosis"), "Aortic Stenosis"),
        ("Mitral", ("mr", "mitral regurgitation", "mitral regurgitant", "mitral insufficiency", "regurgitation"), "Mitral Regurgitation"),
        ("Mitral", ("ms", "mitral stenosis", "mitral stenotic", "stenosis"), "Mitral Stenosis"),
        ("Tricuspid", ("tr", "tricuspid regurgitation", "tricuspid regurgitant", "tricuspid insufficiency", "regurgitation"), "Tricuspid Regurgitation"),
        ("Pulmonary", ("pr", "pulmonary regurgitation", "pulmonic regurgitation", "regurgitation"), "Pulmonary Regurgitation"),
    )
    for valve, tokens, label in valve_checks:
        finding = p.valves.get(valve)
        if finding and (N.is_explicitly_negated(finding, *tokens) or N.is_normal(finding)):
            ruled_out.append(f"{label}: explicitly absent in {valve} Valve Finding")
        elif p.conclusion and N.is_explicitly_negated(p.conclusion, *tokens):
            ruled_out.append(f"{label}: explicitly absent in report conclusion")

    if N.is_explicitly_negated(p.conclusion, "lvoto", "outflow tract obstruction", "systolic anterior motion", "sam"):
        ruled_out.append("LVOTO: explicitly absent in report conclusion")

    if (N.is_explicitly_negated(p.conclusion, "rwma", "regional wall motion abnormality") or
            N.is_explicitly_negated(p.raw.get("rwma"), "rwma", "yes", "present") or
            p.raw.get("rwma") in ("Absent", "Normal", "None", "Nil")):
        ruled_out.append("RWMA: explicitly absent in report")

    if (N.is_explicitly_negated(p.conclusion, "pericardial effusion", "effusion") or
            N.is_explicitly_negated(p.effusion, "pericardial effusion", "effusion", "present", "none/trace") or
            p.effusion in ("None/Trace", "None", "Nil", "Absent", "Normal")):
        ruled_out.append("Pericardial Effusion: explicitly absent in report")

    if (N.is_explicitly_negated(p.conclusion, "thrombus", "clot", "intracardiac clot") or
            N.is_explicitly_negated(p.thrombus, "thrombus", "clot", "present", "no clots") or
            p.thrombus in ("No Clots", "None", "Nil", "Absent", "Normal")):
        ruled_out.append("Intracardiac Thrombus: explicitly absent in report")

    if N.is_explicitly_negated(p.conclusion, "diastolic dysfunction", "impaired relaxation", "restrictive filling", "pseudonormal"):
        ruled_out.append("Diastolic Dysfunction: explicitly absent in report conclusion")

    for name, phrases in _CONCLUSION_ONLY:
        if N.is_explicitly_negated(p.conclusion, *phrases):
            ruled_out.append(f"{name}: explicitly absent in report conclusion")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in ruled_out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


_CLINICAL_SENTINELS = (
    "stenosis", "regurgitation", "regurgitant", "insufficiency", "dysfunction",
    "hypertrophy", "cardiomyopathy", "dilation", "dilatation", "aneurysm",
    "infarction", "ischemia", "ischemic", "thrombus", "clot", "effusion",
    "tamponade", "shunt", "defect", "hypertension", "prolapse", "hypokinesia",
    "akinesia", "dyskinesia", "calcification", "sclerosis", "obstruction",
    "coarctation", "myxoma", "vegetation", "endocarditis", "myocarditis"
)


def _find_unmatched_findings(p: Params, diseases: List[Prediction], ruled_out: List[str]) -> List[Dict[str, str]]:
    """Safety net: detects clinical diagnostic terms in conclusion text that were neither predicted
    by any rule nor negated/ruled out. Flags them for reviewing doctor so nothing is silently lost."""
    if not p.conclusion:
        return []

    clauses = [c.strip() for c in re.split(r"[\n\r;]+|\.(?!\d)", p.conclusion) if c.strip()]
    predicted_text = " ".join([d.name.lower() + " " + " ".join(d.supporting_points).lower() for d in diseases])
    ruled_out_text = " ".join(ruled_out).lower()

    unmatched: List[Dict[str, str]] = []
    seen_clauses = set()

    for clause in clauses:
        low = clause.lower()
        if N.is_normal(clause) or low in seen_clauses:
            continue

        for term in _CLINICAL_SENTINELS:
            match = re.search(r"\b" + re.escape(term) + r"\b", low)
            if match:
                # Check if negated in this clause
                start, end = match.start(), match.end()
                if N.is_negated_match(low, start, end):
                    continue
                # Check if accounted for in predicted diseases or ruled out findings
                if term in predicted_text or term in ruled_out_text:
                    continue
                if any(w in predicted_text for w in low.split() if len(w) > 4):
                    continue

                seen_clauses.add(low)
                unmatched.append({
                    "clause": clause,
                    "term": term,
                    "message": f"Unclassified clinical finding '{clause}' in report conclusion (flagged for review)",
                })
                break
    return unmatched


# ===========================================================================================
# Entry point
# ===========================================================================================
def evaluate_v4(report_params: Dict[str, Any], patient_age: Any = None,
                patient_sex: Any = None,
                field_sources: Optional[Dict[str, str]] = None,
                is_athlete: bool = False) -> Dict[str, Any]:
    """Run the v4.0 engine over one report's stored parameters."""
    band = resolve_age_band(patient_age)
    t = thresholds_for(band)
    age_group, age_unknown = V.resolve_age_group(patient_age)
    is_pediatric = (age_group == V.CHILDREN) or (band == "pediatric")
    sex = V.resolve_sex(patient_sex)
    dropdown_fields = {f for f, src in (field_sources or {}).items() if src == "doctor_dropdown"}
    p = Params(report_params, field_sources=field_sources, is_pediatric=is_pediatric)

    diseases: List[Prediction] = []
    if is_pediatric:
        diseases.append(Prediction(
            "Pediatric Case Review Required", "Pediatric Review", "info",
            ["Pediatric case — numeric thresholds not clinically validated, mandatory manual specialist review required."],
            "Pediatric normal ranges depend on BSA and z-scores -- manual specialist pediatric review required.",
            fields=[]
        ))

    # The rules that read a compulsory group take the age band's thresholds; the rest are
    # age-independent and are called unchanged.
    age_aware = {_rule_ischemia, _rule_hypertrophy, _rule_lvoto}
    for rule in (_rule_thrombus, _rule_ischemia, _rule_lv_function, _rule_pericardium,
                 _rule_cardiomyopathy, _rule_hypertrophy, _rule_pulmonary_pressure,
                 _rule_diastolic, _rule_valves, _rule_lvoto, _rule_chambers, _rule_aorta,
                 _rule_combined, _rule_conclusion_only):
        diseases.extend(rule(p, t) if rule in age_aware else rule(p))

    # De-duplicate by name, keeping the first (highest-priority) occurrence.
    seen, unique = set(), []
    for d in diseases:
        if d.name not in seen:
            seen.add(d.name)
            unique.append(d)
    diseases = sorted(unique, key=lambda d: d.priority)

    # HCM vs Hypertensive Heart Disease co-firing suppression:
    # If HCM is confirmed/predicted (from conclusion text or numeric criteria), suppress generic
    # Hypertensive Heart Disease UNLESS the report conclusion explicitly and separately mentions
    # hypertensive heart disease / hypertensive cardiomyopathy by name.
    has_hcm = any("Hypertrophic Cardiomyopathy" in d.name for d in diseases)
    if has_hcm:
        explicit_hhd = p.says(
            "hypertensive heart disease", "hypertensive cardiomyopathy", "hhd",
            "hypertensive crd", "hypertensive changes"
        )
        if not explicit_hhd:
            diseases = [d for d in diseases if d.name != "Hypertensive Heart Disease"]

    # §9. Diseases only -- deliberately after de-duplication, so a suppressed duplicate can never
    # contribute a second "independent source" to the entry that survived it.
    _apply_scores(diseases, p, age_group, sex, dropdown_fields)

    normal = _rule_normal_heart(p, diseases)
    # §6 -- unconditional differential screen, independent of the self-reported athlete flag.
    gray_zone = _rule_athlete_gray_zone(p, age_group, sex)
    athlete = _rule_athlete(p, diseases, is_athlete, age_group, sex)
    exercise_safety = _rule_exercise_safety(p, t, is_athlete, age_group, sex, gray_zone)
    risk = _risk_score(diseases, normal)
    # §5 -- only meaningful when nothing was predicted; still computed unconditionally so the
    # caller can apply that gate itself rather than re-deriving age_group/sex.
    borderline = _borderline_parameters(p, diseases, age_group, sex) if not diseases else []

    ruled_out_findings = _find_ruled_out_findings(p)
    unmatched_findings = _find_unmatched_findings(p, diseases, ruled_out_findings)
    available = sum(1 for v in p.raw.values() if N.clean(v) is not None)
    return {
        "engine_version": ENGINE_VERSION,
        "diseases": [d.to_dict() for d in diseases],
        "ruled_out_findings": ruled_out_findings,
        "unmatched_clinical_findings": unmatched_findings,
        "normal_heart": normal.to_dict() if normal else None,
        "athlete_screening": athlete.to_dict() if athlete else None,
        "exercise_safety": exercise_safety.to_dict() if exercise_safety else None,
        # §6 -- Athlete's Heart vs HCM gray zone. Runs unconditionally; requires explicit doctor
        # confirmation before use in disease prediction or exercise safety (see the card's own
        # recommendation text). Separate key rather than folded into athlete_screening/
        # exercise_safety so it can be confirmed independently of either verdict.
        "athlete_gray_zone": gray_zone.to_dict() if gray_zone else None,
        "is_athlete": bool(is_athlete),
        "borderline_parameters": borderline,
        "age_band": band,
        "age_band_label": t.label,
        # The master document's four-band age group, separate from the exercise band above. It is
        # what the dropdown value table and the §9 scores were resolved against, so it is reported
        # rather than left implicit.
        "age_group": age_group,
        "age_group_label": V.AGE_GROUPS[age_group]["label"],
        "age_unknown": age_unknown,
        # 0-14: no dropdown option resolved to a value and every entered value is free text.
        "pending_pediatric_review": bool(
            V.AGE_GROUPS[age_group].get("pending_pediatric_review")),
        "pediatric_notice": V.AGE_GROUPS[age_group].get("notice") or None,
        "dropdown_entered_fields": sorted(dropdown_fields),
        "compulsory_coverage": _compulsory_coverage(p),
        "risk": risk.to_dict(),
        "risk_level": risk.name.replace("Overall Risk: ", ""),
        "parameters_available": available,
        "parameters_total": len(PARAM_MAP),
        # Aliases of the two fields above, under the name the review page's completeness line and
        # risk-summary bar read. The API response (schemas.PredictionOut) already renamed them on
        # the way out; this is the SAME dict that gets stored verbatim as report.predicted_diseases
        # and read back unchanged on a page reload, so without these keys here too the count was
        # only ever correct until the next refresh -- it silently read as "0 of 51" after.
        "rules_evaluated": available,
        "rules_total": len(PARAM_MAP),
        "disclaimer": DISCLAIMER,
    }
