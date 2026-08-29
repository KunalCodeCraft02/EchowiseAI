"""
36-rule Cardiac Disease Prediction Engine.

MEDICAL CONSTRAINT: this operates solely as a Clinical Decision Support
System (CDSS). A 2D Echo cannot diagnose coronary artery blockage directly —
Rule 25 is an ischemic *proxy* flag that recommends further workup
(ECG / Stress Test / Coronary Angiography) and must never assert "Heart
Blockage" as a diagnosis. Every prediction requires physician confirmation.

MISSING DATA PROTOCOL (critical):
A missing/unparseable parameter is NEVER treated as 0, False, or "Normal".
Each rule declares exactly which normalized parameters it needs; if any of
them are unavailable, that rule is skipped (`matched=None`, "insufficient
data") while every other rule with enough data still evaluates normally.

Rule thresholds (all lengths in millimeters, PASP in mmHg, EF in %):
  Rule 1       Normal Heart
  Rule 2-4     LV Dysfunction: Mild / Moderate / Severe (Heart Failure)
  Rule 5       Dilated Cardiomyopathy
  Rule 6-7     LV Hypertrophy: Mild / Severe
  Rule 8       Left Atrial Enlargement
  Rule 9       Right Atrial Enlargement
  Rule 10      Right Ventricular Enlargement
  Rule 11      Pulmonary Hypertension
  Rule 12      Grade-I Diastolic Dysfunction
  Rule 13      Grade-II Diastolic Dysfunction (pseudonormal + LA enlargement)
  Rule 14      Advanced (Grade-III) Diastolic Dysfunction
  Rule 15-20   Mitral/Aortic/Tricuspid Regurgitation & Stenosis (graded)
  Rule 21      Pulmonary Valve Disease
  Rule 22      Suspected Right Heart Failure
  Rule 23      Hypertensive Heart Disease
  Rule 24      High Risk Structural Heart Disease
  Rule 25      Possible Ischemic Heart Disease (proxy — recommend workup)
  Rule 26      Athlete: Fit for Sports Clearance
  Rule 27      Athlete: Restrict / Further Evaluation Needed
  Rule 28      Multiple Valve Disease
  Rule 29      Mixed Cardiomyopathy
  Rule 30      Chronic Pressure Overload
  Rule 31-33   Overall Risk Scoring: High / Moderate / Low
  Rule 34      Aortic Stenosis graded from Doppler peak velocity / peak gradient
  Rule 35      Mitral Stenosis graded from mitral valve area
  Rule 36      Elevated Pulmonary Pressure from TR jet velocity

Rules 34-36 grade severity from MEASURED Doppler numbers, where rules 15-21 read the
qualitative finding text. Both are kept deliberately: a report may print either or both, and a
measured gradient is stronger evidence than the word "moderate" in prose.

Execution priority: EF-driven global function -> complex cardiomyopathies ->
structural/wall thickness -> pressures & right heart -> diastolic function ->
valvular disease -> chamber enlargement -> ischemic proxy -> athlete
screening -> overall risk scoring. `evaluate_all()` follows this order.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.predictor import normalize as N

RA_ENLARGED_MM = 44.0   # ASE reference upper limit, RA linear diameter fallback
RV_ENLARGED_MM = 42.0   # ASE reference upper limit, RV basal diameter fallback

DISCLAIMER = (
    "Clinical Decision Support output only — not an automated diagnosis. "
    "A 2D Echo cannot directly diagnose coronary artery blockage. "
    "All findings require physician review and correlation with clinical history."
)


# ---------------------------------------------------------------------------
# Normalized parameter bundle
# ---------------------------------------------------------------------------
@dataclass
class NormalizedParams:
    ef: Optional[float] = None
    lvidd: Optional[float] = None
    lvids: Optional[float] = None
    ivsd: Optional[float] = None
    pwd: Optional[float] = None
    ivss: Optional[float] = None
    pws: Optional[float] = None
    la: Optional[float] = None
    ao: Optional[float] = None
    ivc: Optional[float] = None
    ea: Optional[float] = None
    pasp: Optional[float] = None
    ra_state: Optional[str] = None
    rv_state: Optional[str] = None
    pericardial_effusion: Optional[bool] = None
    clots: Optional[bool] = None
    mv: N.ValveFinding = field(default_factory=lambda: N.ValveFinding(None, None, None))
    av: N.ValveFinding = field(default_factory=lambda: N.ValveFinding(None, None, None))
    tv: N.ValveFinding = field(default_factory=lambda: N.ValveFinding(None, None, None))
    pv: N.ValveFinding = field(default_factory=lambda: N.ValveFinding(None, None, None))
    wall_motion_raw: Optional[str] = None
    # Valve Doppler (Phase 4). Distinct from the mv/av/tv/pv ValveFindings above: those are the
    # qualitative findings, these are the numbers stenosis severity is actually graded on.
    av_vmax: Optional[float] = None      # m/s
    av_peak_grad: Optional[float] = None  # mmHg
    mv_area: Optional[float] = None      # cm2
    tr_vmax: Optional[float] = None      # m/s


def normalize_params(p: Dict[str, Any]) -> NormalizedParams:
    g = p.get
    return NormalizedParams(
        ef=N.parse_percent(g("ef")),
        lvidd=N.parse_length_mm(g("lvidd")),
        lvids=N.parse_length_mm(g("lvids")),
        ivsd=N.parse_length_mm(g("ivsd")),
        pwd=N.parse_length_mm(g("pwd")),
        ivss=N.parse_length_mm(g("ivss")),
        pws=N.parse_length_mm(g("pws")),
        la=N.parse_length_mm(g("la_diameter")),
        ao=N.parse_length_mm(g("ao_diameter")),
        ivc=N.parse_length_mm(g("ivc")),
        ea=N.parse_ratio(g("e_a_ratio")),
        pasp=N.parse_pressure_mmhg(g("pasp")),
        ra_state=N.classify_chamber_size(g("ra_size"), RA_ENLARGED_MM),
        rv_state=N.classify_chamber_size(g("rv_size"), RV_ENLARGED_MM),
        pericardial_effusion=N.classify_presence(g("pericardial_effusion")),
        clots=N.classify_presence(g("clots_thrombus")),
        mv=N.classify_valve(g("mv_finding")),
        av=N.classify_valve(g("av_finding")),
        tv=N.classify_valve(g("tv_finding")),
        pv=N.classify_valve(g("pv_finding")),
        wall_motion_raw=g("wall_motion"),
        av_vmax=N.parse_velocity_ms(g("av_peak_velocity")),
        av_peak_grad=N.parse_pressure_mmhg(g("av_peak_gradient")),
        mv_area=N.parse_area_cm2(g("mv_area")),
        tr_vmax=N.parse_velocity_ms(g("tv_peak_velocity")),
    )


# ---------------------------------------------------------------------------
# Rule result
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    rule_id: int
    name: str
    category: str
    severity: str  # "normal" | "mild" | "moderate" | "severe" | "info"
    matched: Optional[bool]  # True / False / None ("insufficient data")
    supporting_points: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category,
            "severity": self.severity,
            "matched": self.matched,
            "supporting_points": self.supporting_points,
            "recommendation": self.recommendation,
        }


def _fmt_mm(v: float) -> str:
    return f"{v:.0f} mm"


def _fmt_pct(v: float) -> str:
    return f"{v:.0f}%"


def _fmt_ratio(v: float) -> str:
    return f"{v:.2f}"


def _fmt_mmhg(v: float) -> str:
    return f"{v:.0f} mmHg"


def _valve_display(name: str) -> str:
    return {"mv": "Mitral", "av": "Aortic", "tv": "Tricuspid", "pv": "Pulmonary"}[name]


def any_valve_disease(p: NormalizedParams) -> Optional[bool]:
    states = [p.mv.abnormal, p.av.abnormal, p.tv.abnormal, p.pv.abnormal]
    if any(s is True for s in states):
        return True
    if all(s is None for s in states):
        return None
    return False


def diseased_valve_count(p: NormalizedParams) -> Optional[int]:
    states = [p.mv.abnormal, p.av.abnormal, p.tv.abnormal, p.pv.abnormal]
    if all(s is None for s in states):
        return None
    return sum(1 for s in states if s is True)


# ---------------------------------------------------------------------------
# Rules 1-30
# ---------------------------------------------------------------------------
def rule_01_normal_heart(p: NormalizedParams) -> RuleResult:
    r = RuleResult(1, "Normal Heart", "Baseline & Fitness", "normal", None,
                    recommendation="No structural or functional abnormality detected. Routine follow-up as clinically indicated.")
    required = [p.ef, p.ivsd, p.pwd, p.la, p.pasp, p.ea]
    valve_states = [p.mv.abnormal, p.av.abnormal, p.tv.abnormal, p.pv.abnormal]
    if any(v is None for v in required) or any(v is None for v in valve_states):
        return r
    ok = (p.ef >= 55 and p.ivsd <= 11 and p.pwd <= 11 and p.la <= 40
          and p.pasp < 35 and 0.8 <= p.ea <= 2.0 and not any(valve_states))
    r.matched = ok
    if ok:
        r.supporting_points = [
            f"EF {_fmt_pct(p.ef)} (>=55%)", f"IVSd {_fmt_mm(p.ivsd)} (<=11mm)", f"PWd {_fmt_mm(p.pwd)} (<=11mm)",
            f"LA {_fmt_mm(p.la)} (<=40mm)", f"PASP {_fmt_mmhg(p.pasp)} (<35mmHg)",
            f"E/A {_fmt_ratio(p.ea)} (0.8-2.0)", "All valves Normal",
        ]
    return r


def rule_02_04_lv_dysfunction(p: NormalizedParams) -> List[RuleResult]:
    mild = RuleResult(2, "Mild LV Dysfunction", "LV Dysfunction & Ischemia", "mild", None,
                       recommendation="Mild systolic impairment. Guideline-directed lifestyle measures; reassess EF on interval echo.")
    mod = RuleResult(3, "Moderate LV Dysfunction", "LV Dysfunction & Ischemia", "moderate", None,
                      recommendation="Moderate systolic impairment. Cardiology referral for guideline-directed medical therapy (GDMT) evaluation.")
    sev = RuleResult(4, "Severe LV Dysfunction (Heart Failure)", "LV Dysfunction & Ischemia", "severe", None,
                      recommendation="Severe systolic impairment consistent with heart failure with reduced EF. Urgent cardiology evaluation and GDMT initiation/titration.")
    if p.ef is None:
        return [mild, mod, sev]
    pt = f"EF {_fmt_pct(p.ef)}"
    mild.matched = 45 <= p.ef <= 54
    mod.matched = 35 <= p.ef <= 44
    sev.matched = p.ef < 35
    if mild.matched:
        mild.supporting_points = [f"{pt} (45-54% range)"]
    if mod.matched:
        mod.supporting_points = [f"{pt} (35-44% range)"]
    if sev.matched:
        sev.supporting_points = [f"{pt} (<35%)"]
    return [mild, mod, sev]


def rule_05_dilated_cardiomyopathy(p: NormalizedParams) -> RuleResult:
    r = RuleResult(5, "Dilated Cardiomyopathy", "LV Dysfunction & Ischemia", "severe", None,
                    recommendation="Pattern consistent with dilated cardiomyopathy. Cardiology referral, etiology workup (ischemic vs non-ischemic), and GDMT.")
    if p.ef is None or p.lvidd is None or p.lvids is None:
        return r
    r.matched = p.ef < 45 and p.lvidd > 58 and p.lvids > 40
    if r.matched:
        r.supporting_points = [f"EF {_fmt_pct(p.ef)} (<45%)", f"LVIDd {_fmt_mm(p.lvidd)} (>58mm)", f"LVIDs {_fmt_mm(p.lvids)} (>40mm)"]
    return r


def _thickest_wall(p: NormalizedParams):
    vals = [v for v in (p.ivsd, p.pwd) if v is not None]
    if not vals:
        return None, []
    thick = max(vals)
    pts = []
    if p.ivsd is not None:
        pts.append(f"IVSd {_fmt_mm(p.ivsd)}")
    if p.pwd is not None:
        pts.append(f"PWd {_fmt_mm(p.pwd)}")
    return thick, pts


def rule_06_07_lv_hypertrophy(p: NormalizedParams) -> List[RuleResult]:
    mild = RuleResult(6, "Mild LV Hypertrophy", "Hypertrophy & Structural", "mild", None,
                       recommendation="Mild wall thickening. Screen/optimize blood pressure control; reassess periodically.")
    sev = RuleResult(7, "Severe LV Hypertrophy", "Hypertrophy & Structural", "severe", None,
                      recommendation="Severe wall thickening. Cardiology referral to evaluate for hypertensive heart disease vs. hypertrophic cardiomyopathy.")
    thick, pts = _thickest_wall(p)
    if thick is None:
        return [mild, sev]
    mild.matched = 11 < thick <= 14
    sev.matched = thick > 14
    if mild.matched:
        mild.supporting_points = pts + ["Thickest wall in mild range (11-14mm)"]
    if sev.matched:
        sev.supporting_points = pts + ["Thickest wall severe range (>14mm)"]
    return [mild, sev]


def rule_08_la_enlargement(p: NormalizedParams) -> RuleResult:
    r = RuleResult(8, "Left Atrial Enlargement", "Chambers & Pressures", "mild", None,
                    recommendation="LA enlargement — evaluate for chronic volume/pressure overload, AF risk, and diastolic dysfunction.")
    if p.la is None:
        return r
    r.matched = p.la > 40
    if r.matched:
        r.supporting_points = [f"LA {_fmt_mm(p.la)} (>40mm)"]
    return r


def rule_09_ra_enlargement(p: NormalizedParams) -> RuleResult:
    r = RuleResult(9, "Right Atrial Enlargement", "Chambers & Pressures", "mild", None,
                    recommendation="RA enlargement — evaluate right heart pressures and tricuspid valve function.")
    if p.ra_state is None:
        return r
    r.matched = p.ra_state == "Enlarged"
    if r.matched:
        r.supporting_points = ["RA Size: Enlarged"]
    return r


def rule_10_rv_enlargement(p: NormalizedParams) -> RuleResult:
    r = RuleResult(10, "Right Ventricular Enlargement", "Chambers & Pressures", "mild", None,
                    recommendation="RV enlargement — evaluate for pulmonary hypertension, RV volume/pressure overload.")
    if p.rv_state is None:
        return r
    r.matched = p.rv_state == "Enlarged"
    if r.matched:
        r.supporting_points = ["RV Size: Enlarged"]
    return r


def rule_11_pulmonary_hypertension(p: NormalizedParams) -> RuleResult:
    r = RuleResult(11, "Pulmonary Hypertension", "Chambers & Pressures", "moderate", None,
                    recommendation="Elevated estimated PASP — correlate clinically; consider right heart catheterization if indicated.")
    if p.pasp is None:
        return r
    r.matched = p.pasp > 35
    if r.matched:
        r.supporting_points = [f"PASP {_fmt_mmhg(p.pasp)} (>35mmHg)"]
    return r


def rule_12_diastolic_grade1(p: NormalizedParams) -> RuleResult:
    r = RuleResult(12, "Grade-I Diastolic Dysfunction", "Diastolic Function", "mild", None,
                    recommendation="Impaired relaxation pattern. Manage contributing risk factors (BP, weight, glycemic control).")
    if p.ea is None:
        return r
    r.matched = p.ea < 0.8
    if r.matched:
        r.supporting_points = [f"E/A {_fmt_ratio(p.ea)} (<0.8)"]
    return r


def rule_13_diastolic_grade2(p: NormalizedParams) -> RuleResult:
    r = RuleResult(13, "Grade-II Diastolic Dysfunction", "Diastolic Function", "moderate", None,
                    recommendation="Pseudonormal filling pattern with LA enlargement. Cardiology evaluation for diastolic heart failure risk.")
    if p.ea is None or p.la is None:
        return r
    r.matched = 0.8 <= p.ea <= 2.0 and p.la > 40
    if r.matched:
        r.supporting_points = [f"E/A {_fmt_ratio(p.ea)} (0.8-2.0)", f"LA {_fmt_mm(p.la)} (>40mm)"]
    return r


def rule_14_diastolic_advanced(p: NormalizedParams) -> RuleResult:
    r = RuleResult(14, "Advanced (Grade-III) Diastolic Dysfunction", "Diastolic Function", "severe", None,
                    recommendation="Restrictive filling pattern. Urgent cardiology evaluation for advanced diastolic/heart failure.")
    if p.ea is None:
        return r
    r.matched = p.ea > 2.0
    if r.matched:
        r.supporting_points = [f"E/A {_fmt_ratio(p.ea)} (>2.0)"]
    return r


_VALVE_RULE_SPECS = [
    (15, "mv", "regurgitation", "Mitral Regurgitation"),
    (16, "mv", "stenosis", "Mitral Stenosis"),
    (17, "av", "regurgitation", "Aortic Regurgitation"),
    (18, "av", "stenosis", "Aortic Stenosis"),
    (19, "tv", "regurgitation", "Tricuspid Regurgitation"),
    (20, "tv", "stenosis", "Tricuspid Stenosis"),
]

_SEVERITY_LABEL = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe"}


def rule_15_20_valve_disease(p: NormalizedParams) -> List[RuleResult]:
    results = []
    for rule_id, field_name, target_kind, base_name in _VALVE_RULE_SPECS:
        valve: N.ValveFinding = getattr(p, field_name)
        sev = valve.severity or "moderate"  # unspecified severity is treated conservatively as moderate for display
        display_name = f"{_SEVERITY_LABEL.get(valve.severity, 'Unspecified-Severity')} {base_name}"
        r = RuleResult(rule_id, display_name, "Valvular Diseases", sev, None,
                        recommendation=f"{base_name} detected. Cardiology referral for serial echo surveillance and symptom assessment; "
                                       f"intervention timing per severity and symptoms.")
        if valve.abnormal is None:
            results.append(r)
            continue
        r.matched = valve.abnormal and valve.kind == target_kind
        if r.matched:
            raw_field = {"mv": "MV", "av": "AV", "tv": "TV"}[field_name]
            r.supporting_points = [f"{raw_field} finding: {base_name}" + (f" ({valve.severity})" if valve.severity else " (severity not specified in report)")]
        results.append(r)
    return results


def rule_21_pulmonary_valve(p: NormalizedParams) -> RuleResult:
    r = RuleResult(21, "Pulmonary Valve Disease", "Valvular Diseases", "mild", None,
                    recommendation="Pulmonary valve abnormality noted. Correlate clinically; routine cardiology follow-up.")
    if p.pv.abnormal is None:
        return r
    r.matched = p.pv.abnormal is True
    if r.matched:
        r.supporting_points = ["PV finding: abnormal (non-normal state)"]
    return r


def rule_22_right_heart_failure(p: NormalizedParams) -> RuleResult:
    r = RuleResult(22, "Suspected Right Heart Failure", "Chambers & Pressures", "severe", None,
                    recommendation="Combined RV/RA enlargement with elevated PASP. Urgent cardiology evaluation for right heart failure.")
    if p.rv_state is None or p.ra_state is None or p.pasp is None:
        return r
    r.matched = p.rv_state == "Enlarged" and p.ra_state == "Enlarged" and p.pasp > 35
    if r.matched:
        r.supporting_points = ["RV Size: Enlarged", "RA Size: Enlarged", f"PASP {_fmt_mmhg(p.pasp)} (>35mmHg)"]
    return r


def rule_23_hypertensive_hd(p: NormalizedParams) -> RuleResult:
    r = RuleResult(23, "Hypertensive Heart Disease", "Hypertrophy & Structural", "moderate", None,
                    recommendation="Concentric wall thickening with LA enlargement, consistent with chronic hypertensive changes. Optimize BP control.")
    if p.ivsd is None or p.pwd is None or p.la is None:
        return r
    r.matched = p.ivsd > 11 and p.pwd > 11 and p.la > 40
    if r.matched:
        r.supporting_points = [f"IVSd {_fmt_mm(p.ivsd)} (>11mm)", f"PWd {_fmt_mm(p.pwd)} (>11mm)", f"LA {_fmt_mm(p.la)} (>40mm)"]
    return r


def rule_24_high_risk_structural(p: NormalizedParams) -> RuleResult:
    r = RuleResult(24, "High Risk Structural Heart Disease", "Hypertrophy & Structural", "severe", None,
                    recommendation="Severe systolic dysfunction with marked structural remodeling and elevated PASP. Urgent cardiology evaluation.")
    if p.ef is None or p.ivsd is None or p.la is None or p.pasp is None:
        return r
    r.matched = p.ef < 35 and p.ivsd > 14 and p.la > 45 and p.pasp > 40
    if r.matched:
        r.supporting_points = [f"EF {_fmt_pct(p.ef)} (<35%)", f"IVSd {_fmt_mm(p.ivsd)} (>14mm)",
                                f"LA {_fmt_mm(p.la)} (>45mm)", f"PASP {_fmt_mmhg(p.pasp)} (>40mmHg)"]
    return r


def rule_25_possible_ischemic_hd(p: NormalizedParams) -> RuleResult:
    r = RuleResult(25, "Possible Ischemic Heart Disease (proxy)", "LV Dysfunction & Ischemia", "moderate", None,
                    recommendation=("Reduced EF with LV systolic dilation but normal wall thickness — a pattern that can occur with "
                                     "ischemic injury. This is a screening proxy only: recommend ECG, stress testing, and coronary "
                                     "angiography for definitive evaluation. A 2D Echo cannot diagnose coronary artery blockage directly."))
    if p.ef is None or p.lvids is None or p.ivsd is None or p.pwd is None:
        return r
    r.matched = p.ef < 45 and p.lvids > 40 and p.ivsd <= 11 and p.pwd <= 11
    if r.matched:
        r.supporting_points = [f"EF {_fmt_pct(p.ef)} (<45%)", f"LVIDs {_fmt_mm(p.lvids)} (>40mm)",
                                f"IVSd {_fmt_mm(p.ivsd)} (<=11mm, non-hypertrophied)", f"PWd {_fmt_mm(p.pwd)} (<=11mm, non-hypertrophied)"]
    return r


def rule_29_mixed_cardiomyopathy(p: NormalizedParams) -> RuleResult:
    r = RuleResult(29, "Mixed Cardiomyopathy", "LV Dysfunction & Ischemia", "severe", None,
                    recommendation="Combined hypertrophic and dilated features with reduced EF. Cardiology referral for etiology workup and GDMT.")
    if p.ef is None or p.ivsd is None or p.pwd is None or p.lvidd is None:
        return r
    r.matched = p.ef < 45 and p.ivsd > 11 and p.pwd > 11 and p.lvidd > 58
    if r.matched:
        r.supporting_points = [f"EF {_fmt_pct(p.ef)} (<45%)", f"IVSd {_fmt_mm(p.ivsd)} (>11mm)",
                                f"PWd {_fmt_mm(p.pwd)} (>11mm)", f"LVIDd {_fmt_mm(p.lvidd)} (>58mm)"]
    return r


def rule_30_chronic_pressure_overload(p: NormalizedParams) -> RuleResult:
    r = RuleResult(30, "Chronic Pressure Overload", "Chambers & Pressures", "moderate", None,
                    recommendation="LA enlargement with elevated PASP suggests chronic pressure/volume overload. Cardiology evaluation recommended.")
    if p.la is None or p.pasp is None:
        return r
    r.matched = p.la > 40 and p.pasp > 35
    if r.matched:
        r.supporting_points = [f"LA {_fmt_mm(p.la)} (>40mm)", f"PASP {_fmt_mmhg(p.pasp)} (>35mmHg)"]
    return r


def rule_28_multiple_valve_disease(p: NormalizedParams) -> RuleResult:
    r = RuleResult(28, "Multiple Valve Disease", "Valvular Diseases", "severe", None,
                    recommendation="Two or more diseased valves detected. Cardiology referral for comprehensive valvular assessment.")
    count = diseased_valve_count(p)
    if count is None:
        return r
    r.matched = count >= 2
    if r.matched:
        names = []
        for field_name in ("mv", "av", "tv", "pv"):
            vf: N.ValveFinding = getattr(p, field_name)
            if vf.abnormal:
                names.append(_valve_display(field_name))
        r.supporting_points = [f"Diseased valves ({count}): {', '.join(names)}"]
    return r


def rule_26_27_athlete_screening(p: NormalizedParams, normal_heart_matched: Optional[bool]) -> List[RuleResult]:
    fit = RuleResult(26, "Athlete: Fit for Sports Clearance", "Baseline & Fitness", "normal", None,
                      recommendation="Echo parameters meet normal-heart criteria. No echo-based contraindication to competitive sports identified.")
    restrict = RuleResult(27, "Athlete: Restrict / Further Evaluation Needed", "Baseline & Fitness", "moderate", None,
                           recommendation="One or more parameters outside normal range for sports clearance. Refer to sports cardiology before clearance.")

    fit.matched = normal_heart_matched
    if fit.matched:
        fit.supporting_points = ["Meets all Normal Heart criteria"]

    branches = [
        (p.ef is not None and p.ef < 55, f"EF {_fmt_pct(p.ef)} (<55%)" if p.ef is not None else None),
        (p.ivsd is not None and p.ivsd > 11, f"IVSd {_fmt_mm(p.ivsd)} (>11mm)" if p.ivsd is not None else None),
        (p.pwd is not None and p.pwd > 11, f"PWd {_fmt_mm(p.pwd)} (>11mm)" if p.pwd is not None else None),
        (p.pasp is not None and p.pasp > 35, f"PASP {_fmt_mmhg(p.pasp)} (>35mmHg)" if p.pasp is not None else None),
    ]
    valve_disease = any_valve_disease(p)
    if valve_disease is True:
        branches.append((True, "Valve disease present"))
    elif valve_disease is False:
        branches.append((False, None))

    known_branches = [(hit, pt) for hit, pt in branches if pt is not None or hit]
    any_data = any(v is not None for v in (p.ef, p.ivsd, p.pwd, p.pasp)) or valve_disease is not None
    if not any_data:
        return [fit, restrict]

    hits = [pt for hit, pt in branches if hit and pt]
    restrict.matched = len(hits) > 0
    if restrict.matched:
        restrict.supporting_points = hits
    return [fit, restrict]


# ---------------------------------------------------------------------------
# Rules 31-33: overall risk scoring
# ---------------------------------------------------------------------------
DISEASE_RULE_IDS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 28, 29, 30}


def rules_31_33_risk_scoring(disease_results: List[RuleResult]) -> List[RuleResult]:
    high = RuleResult(31, "Overall Risk: High", "Overall Risk Scoring", "severe", None,
                       recommendation="Multiple concurrent cardiac abnormalities detected. Prioritize urgent cardiology referral.")
    mod = RuleResult(32, "Overall Risk: Moderate", "Overall Risk Scoring", "moderate", None,
                      recommendation="Two concurrent findings, or a single non-mild finding. Cardiology follow-up recommended.")
    low = RuleResult(33, "Overall Risk: Low", "Overall Risk Scoring", "mild", None,
                      recommendation="A single mild finding. Routine follow-up as clinically indicated.")

    scored = [r for r in disease_results if r.rule_id in DISEASE_RULE_IDS]
    evaluated = [r for r in scored if r.matched is not None]
    if not evaluated:
        return [high, mod, low]  # all None — insufficient data to score risk

    matched = [r for r in scored if r.matched is True]
    count = len(matched)

    if count >= 3:
        high.matched = True
        high.supporting_points = [f"{len(matched)} concurrent findings: " + ", ".join(r.name for r in matched)]
        mod.matched = False
        low.matched = False
    elif count == 2:
        mod.matched = True
        mod.supporting_points = [f"2 concurrent findings: " + ", ".join(r.name for r in matched)]
        high.matched = False
        low.matched = False
    elif count == 1:
        only = matched[0]
        if only.severity == "mild":
            low.matched = True
            low.supporting_points = [f"Single mild finding: {only.name}"]
            high.matched, mod.matched = False, False
        else:
            # A single non-mild finding is elevated to Moderate rather than
            # under-stated as "low risk" (safety-conservative extension).
            mod.matched = True
            mod.supporting_points = [f"Single {only.severity} finding: {only.name}"]
            high.matched, low.matched = False, False
    else:
        high.matched, mod.matched, low.matched = False, False, False

    return [high, mod, low]


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Rules 34-36: quantitative valve grading (Phase 4)
#
# These grade severity from the MEASURED Doppler numbers rather than from the qualitative
# finding text that rules 15-21 read. Both are kept: a report may print one, the other, or
# both, and a measured gradient is stronger evidence than the word "moderate" in prose.
# Thresholds are the standard ASE/ESC valve-grading cut-points.
# ---------------------------------------------------------------------------
def rule_34_aortic_stenosis_severity(p: NormalizedParams) -> RuleResult:
    r = RuleResult(34, "Aortic Stenosis (by Doppler)", "Valves", "moderate", None,
                   recommendation="Quantitative aortic stenosis grading — correlate with valve "
                                  "area and symptoms; severe AS warrants prompt cardiology review.")
    if p.av_vmax is None and p.av_peak_grad is None:
        return r

    # Velocity is the primary grader; peak gradient is used when velocity is absent.
    severity = None
    points: List[str] = []
    if p.av_vmax is not None:
        if p.av_vmax >= 4.0:
            severity, sev_word = "severe", "severe"
        elif p.av_vmax >= 3.0:
            severity, sev_word = "moderate", "moderate"
        elif p.av_vmax >= 2.6:
            severity, sev_word = "mild", "mild"
        if severity:
            points.append(f"AV peak velocity {p.av_vmax:.2f} m/s ({sev_word} range)")
    if severity is None and p.av_peak_grad is not None:
        if p.av_peak_grad > 64:
            severity = "severe"
        elif p.av_peak_grad >= 36:
            severity = "moderate"
        elif p.av_peak_grad >= 20:
            severity = "mild"
        if severity:
            points.append(f"AV peak gradient {_fmt_mmhg(p.av_peak_grad)} ({severity} range)")
    elif severity and p.av_peak_grad is not None:
        points.append(f"AV peak gradient {_fmt_mmhg(p.av_peak_grad)}")

    r.matched = severity is not None
    if r.matched:
        r.severity = severity
        r.name = f"Aortic Stenosis — {severity.capitalize()} (by Doppler)"
        r.supporting_points = points
    return r


def rule_35_mitral_stenosis_severity(p: NormalizedParams) -> RuleResult:
    r = RuleResult(35, "Mitral Stenosis (by Valve Area)", "Valves", "moderate", None,
                   recommendation="Quantitative mitral stenosis grading — correlate with gradient, "
                                  "PA pressure and symptoms; severe MS warrants intervention review.")
    if p.mv_area is None:
        return r

    if p.mv_area < 1.0:
        severity = "severe"
    elif p.mv_area < 1.5:
        severity = "moderate"
    elif p.mv_area <= 2.0:
        severity = "mild"
    else:
        severity = None

    r.matched = severity is not None
    if r.matched:
        r.severity = severity
        r.name = f"Mitral Stenosis — {severity.capitalize()} (by Valve Area)"
        r.supporting_points = [f"Mitral valve area {p.mv_area:.2f} cm2 ({severity} range)"]
    return r


def rule_36_tr_velocity_pulmonary_pressure(p: NormalizedParams) -> RuleResult:
    r = RuleResult(36, "Elevated Pulmonary Pressure (by TR Velocity)", "Chambers & Pressures",
                   "moderate", None,
                   recommendation="TR jet velocity suggests raised pulmonary artery pressure — "
                                  "correlate with RV size/function and clinical context.")
    if p.tr_vmax is None:
        return r

    r.matched = p.tr_vmax > 2.8
    if r.matched:
        r.severity = "severe" if p.tr_vmax > 3.4 else "moderate"
        r.supporting_points = [f"TR peak velocity {p.tr_vmax:.2f} m/s (>2.8 m/s)"]
        if p.pasp is not None:
            r.supporting_points.append(f"Reported PASP {_fmt_mmhg(p.pasp)}")
    return r


def evaluate_all(report_params: Dict[str, Any]) -> Dict[str, Any]:
    p = normalize_params(report_params)

    normal_heart = rule_01_normal_heart(p)
    lv_dysfunction = rule_02_04_lv_dysfunction(p)
    dilated_cmp = rule_05_dilated_cardiomyopathy(p)
    mixed_cmp = rule_29_mixed_cardiomyopathy(p)
    lvh = rule_06_07_lv_hypertrophy(p)
    hypertensive_hd = rule_23_hypertensive_hd(p)
    high_risk_structural = rule_24_high_risk_structural(p)
    pulm_htn = rule_11_pulmonary_hypertension(p)
    right_hf = rule_22_right_heart_failure(p)
    pressure_overload = rule_30_chronic_pressure_overload(p)
    diastolic = [rule_12_diastolic_grade1(p), rule_13_diastolic_grade2(p), rule_14_diastolic_advanced(p)]
    valves = rule_15_20_valve_disease(p)
    pv_disease = rule_21_pulmonary_valve(p)
    multi_valve = rule_28_multiple_valve_disease(p)
    doppler_valves = [rule_34_aortic_stenosis_severity(p), rule_35_mitral_stenosis_severity(p),
                      rule_36_tr_velocity_pulmonary_pressure(p)]
    chamber = [rule_08_la_enlargement(p), rule_09_ra_enlargement(p), rule_10_rv_enlargement(p)]
    ischemic_proxy = rule_25_possible_ischemic_hd(p)
    athlete = rule_26_27_athlete_screening(p, normal_heart.matched)

    all_disease_rules: List[RuleResult] = (
        [normal_heart] + lv_dysfunction + [dilated_cmp, mixed_cmp]
        + lvh + [hypertensive_hd, high_risk_structural, pulm_htn, right_hf, pressure_overload]
        + diastolic + valves + [pv_disease, multi_valve] + chamber + [ischemic_proxy]
        + doppler_valves
    )

    risk = rules_31_33_risk_scoring(all_disease_rules)

    everything = all_disease_rules + athlete + risk
    rules_evaluated = sum(1 for r in everything if r.matched is not None)

    diseases = [r.to_dict() for r in all_disease_rules if r.matched is True and r.rule_id != 1]
    normal_heart_hit = normal_heart.to_dict() if normal_heart.matched is True else None

    matched_risk = next((r for r in risk if r.matched is True), None)
    risk_level = None
    if matched_risk is not None:
        risk_level = matched_risk.name.replace("Overall Risk: ", "")
    elif normal_heart.matched is True:
        risk_level = "Normal"
    elif rules_evaluated == 0:
        risk_level = "Insufficient Data"
    else:
        risk_level = "Indeterminate"

    athlete_hit = next((r.to_dict() for r in athlete if r.matched is True), None)

    return {
        "risk_level": risk_level,
        "rules_total": 36,
        "rules_evaluated": rules_evaluated,
        "normal_heart": normal_heart_hit,
        "diseases": diseases,
        "athlete_screening": athlete_hit,
        "risk_summary": matched_risk.to_dict() if matched_risk else None,
        "disclaimer": DISCLAIMER,
    }
