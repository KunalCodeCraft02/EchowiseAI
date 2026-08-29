"""
Tests for the v4.0 Cardiac Disease Prediction Rule Engine.

The property that matters most here is NOT "does it find diseases" -- it is "does it stay silent
when it does not know". A CDSS that turns missing data into a normal result is worse than no
CDSS at all, so most of these assert absence.
"""
import pytest

from app.predictor import normalize_v4 as N
from app.predictor import rules_v4 as R
from app.predictor.rules_v4 import PARAM_MAP, PRIORITY, evaluate_v4


def names(result):
    return [d["cardiac_disease_name"] for d in result["diseases"]]


# ===========================================================================================
# G1 -- missing data is never imputed
# ===========================================================================================
@pytest.mark.parametrize("missing", [
    None, "", "  ", "not in report", "Not Detected", "not mentioned", "enter manually",
    "N/A", "nil", "--", "unknown",
])
def test_every_missing_token_reads_as_absent(missing):
    assert N.clean(missing) is None
    assert N.to_mm(missing) is None
    assert N.to_percent(missing) is None


def test_empty_report_predicts_nothing_and_is_not_normal():
    """An empty report is 'we don't know', never 'this heart is fine'."""
    out = evaluate_v4({})
    assert out["diseases"] == []
    assert out["normal_heart"] is None
    assert out["athlete_screening"] is None
    assert out["risk_level"] == "Insufficient Data"


def test_normal_heart_needs_positive_evidence_not_just_absence_of_disease():
    """Two normal values are not enough to certify a normal heart."""
    assert evaluate_v4({"ef": "60%", "lvidd": "48 mm"})["normal_heart"] is None


def test_normal_heart_fires_when_broadly_measured_and_all_within_limits():
    out = evaluate_v4({
        "ef": "62%", "lvidd": "48 mm", "lvids": "30 mm", "ivsd": "9 mm", "pwd": "9 mm",
        "la_diameter": "34 mm", "pasp": "22 mmHg", "ra_size": "Normal", "rv_size": "Normal",
        "av_finding": "Normal", "mv_finding": "Normal", "tv_finding": "Normal",
        "pv_finding": "Normal",
    })
    assert out["normal_heart"]["cardiac_disease_name"] == "Normal Heart"
    assert out["diseases"] == []


# ===========================================================================================
# Magnitude-based unit inference
# ===========================================================================================
@pytest.mark.parametrize("raw,expected", [
    ("4.31 cm", 43.1), ("43.1 mm", 43.1),
    ("6.2", 62.0),          # unitless 1.0-9.9 -> cm
    ("53", 53.0),           # unitless 10-99  -> already mm
    ("250", None),          # >100 is biologically impossible -> discarded, never clamped
])
def test_chamber_dimension_unit_inference(raw, expected):
    result = N.to_mm(raw)
    assert result is None if expected is None else result == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    ("1.4 m/s", 1.4), ("134.9 cm/s", 1.349), ("2.6", 2.6), ("45", None),
])
def test_doppler_velocity_inference(raw, expected):
    assert (N.to_velocity_ms(raw) is None if expected is None
            else N.to_velocity_ms(raw) == pytest.approx(expected))


def test_gradient_and_velocity_bands_do_not_overlap():
    """A unitless 45 is a gradient, not a 45 m/s velocity."""
    assert N.to_gradient_mmhg("45") == 45.0
    assert N.to_velocity_ms("45") is None


# ===========================================================================================
# G4 -- synonym folding
# ===========================================================================================
@pytest.mark.parametrize("raw,folded", [
    ("Good LV Function", "normal ef"), ("Poor LV Function", "lv dysfunction"),
    ("Hypokinesia", "hypokinetic"), ("Akinesia", "akinetic"),
    ("Enlarged", "dilated"), ("Dilatation", "dilated"),
    ("Trace", "mild"), ("Physiological", "mild"),
    ("Plethoric IVC", "raised right atrial pressure"),
])
def test_synonym_dictionary(raw, folded):
    assert N.keyword(raw) == folded


# ===========================================================================================
# Rule behaviour
# ===========================================================================================
@pytest.mark.parametrize("ef,expected", [
    ("50%", "Mild Left Ventricular Dysfunction"),
    ("40%", "Moderate Left Ventricular Dysfunction"),
    ("30%", "Severe Left Ventricular Dysfunction"),
])
def test_lv_dysfunction_grading(ef, expected):
    assert expected in names(evaluate_v4({"ef": ef}))


def test_hfref_threshold():
    assert "Heart Failure with Reduced Ejection Fraction (HFrEF)" in names(evaluate_v4({"ef": "35%"}))
    assert "Heart Failure with Reduced Ejection Fraction (HFrEF)" not in names(evaluate_v4({"ef": "45%"}))


def test_dcm_requires_all_three_criteria():
    full = {"ef": "40%", "lvidd": "62 mm", "lvids": "45 mm"}
    assert "Dilated Cardiomyopathy (DCM)" in names(evaluate_v4(full))
    # One criterion short -> must not fire off the other two.
    assert "Dilated Cardiomyopathy (DCM)" not in names(evaluate_v4({**full, "lvidd": "50 mm"}))


def test_hcm_guardrail_never_fires_from_thickness_alone():
    """The spec's explicit guardrail: HCM is never diagnosed from measurements alone."""
    thick_only = names(evaluate_v4({"ivsd": "16 mm"}))
    assert not any("Hypertrophic" in n for n in thick_only)

    # Thick septum WITH preserved EF is reported only as POSSIBLE.
    possible = names(evaluate_v4({"ivsd": "16 mm", "ef": "60%"}))
    assert "Possible Hypertrophic Cardiomyopathy" in possible
    assert "Hypertrophic Cardiomyopathy" not in possible

    # Only the conclusion can assert it outright.
    stated = names(evaluate_v4({"impression_text": "Hypertrophic Cardiomyopathy"}))
    assert "Hypertrophic Cardiomyopathy" in stated


@pytest.mark.parametrize("pasp,grade", [("40 mmHg", "Mild"), ("50 mmHg", "Moderate"), ("70 mmHg", "Severe")])
def test_pulmonary_hypertension_grading(pasp, grade):
    assert f"Pulmonary Hypertension ({grade})" in names(evaluate_v4({"pasp": pasp}))


@pytest.mark.parametrize("ea,la,expected", [
    ("0.6", None, "Grade I Diastolic Dysfunction (Impaired Relaxation)"),
    ("1.2", "44 mm", "Grade II Diastolic Dysfunction (Pseudonormal)"),
    ("2.4", None, "Grade III Diastolic Dysfunction (Restrictive Filling)"),
])
def test_diastolic_grading(ea, la, expected):
    params = {"e_a_ratio": ea}
    if la:
        params["la_diameter"] = la
    assert expected in names(evaluate_v4(params))


def test_regional_wall_motion_triggers_ischemia_proxy():
    out = evaluate_v4({"septal_wall_motion": "Hypokinetic"})
    assert "Regional Wall Motion Abnormality (RWMA)" in names(out)
    cad = next(d for d in out["diseases"] if "Ischemic" in d["cardiac_disease_name"])
    assert "PROXY" in cad["recommendation"]      # never asserted as a coronary diagnosis


def test_mixed_valve_disease_when_one_valve_has_both_lesions():
    out = evaluate_v4({"mv_finding": "Moderate mitral stenosis with mild regurgitation"})
    assert "Mixed Mitral Valve Disease" in names(out)


def test_multiple_valve_disease_needs_two_valves():
    # 2 valves at Moderate or worse -> Multiple Valve Disease fires
    two_mod = evaluate_v4({"mv_finding": "Moderate regurgitation", "av_finding": "Moderate stenosis"})
    assert "Multiple Valve Disease" in names(two_mod)
    # Mild on one and Moderate on another -> does not fire Multiple Valve Disease
    one_mild_one_mod = evaluate_v4({"mv_finding": "Mild regurgitation", "av_finding": "Moderate stenosis"})
    assert "Multiple Valve Disease" not in names(one_mild_one_mod)
    # Mild across 2 valves -> does not fire Multiple Valve Disease
    two_mild = evaluate_v4({"mv_finding": "Mild regurgitation", "tv_finding": "Mild regurgitation"})
    assert "Multiple Valve Disease" not in names(two_mild)


# ===========================================================================================
# G3 -- conclusion text
# ===========================================================================================
def test_conclusion_alone_can_trigger_a_disease():
    out = evaluate_v4({"impression_text": "CONCLUSION: Moderate LV Dysfunction."})
    hit = next(d for d in out["diseases"]
               if d["cardiac_disease_name"] == "Moderate Left Ventricular Dysfunction")
    assert "Conclusion text explicitly mentioned" in hit["supporting_points"][0]


def test_severity_is_read_next_to_its_own_phrase():
    """A conclusion grades several things at once. "Mild MR" must not inherit the "Moderate"
    from a different sentence."""
    out = evaluate_v4({"impression_text": "Moderate LV Dysfunction. Mild MR. Severe Aortic Stenosis."})
    assert "Mild Mitral Regurgitation" in names(out)
    assert "Severe Aortic Stenosis" in names(out)


def test_conclusion_evidence_quotes_the_reports_own_casing():
    out = evaluate_v4({"impression_text": "Mild MR noted."})
    hit = next(d for d in out["diseases"] if "Mitral Regurgitation" in d["cardiac_disease_name"])
    assert "'MR'" in hit["supporting_points"][0]      # not 'mr'


def test_normal_valves_conclusion_does_not_trigger_regurgitation_false_positives():
    """Section 7 Item 2 regression: 2-letter tokens ('ar', 'tr', 'pr') must not match
    inside 'trileaflet', 'pressure', etc., and explicitly normal valves must be suppressed."""
    conclusion = (
        "...Normal, trileaflet aortic valve. Normal mitral valve. "
        "Normal tricuspid valve. Normal PA pressure. Normal pulmonic valve..."
    )
    out = evaluate_v4({"impression_text": conclusion})
    flagged = names(out)
    assert not any("Aortic Regurgitation" in n for n in flagged)
    assert not any("Tricuspid Regurgitation" in n for n in flagged)
    assert not any("Pulmonary Regurgitation" in n for n in flagged)
    assert not any("Mitral Regurgitation" in n for n in flagged)


def test_substring_collisions_in_prose_do_not_trigger_regurgitation():
    """Words like 'artery', 'trileaflet', 'compared', 'previous', 'pressure', 'primary'
    must not match 'ar', 'tr', 'pr'."""
    conclusion = (
        "Compared with previous report, coronary artery dimensions and PA pressure are normal. "
        "Primary cardiac structures intact."
    )
    out = evaluate_v4({"impression_text": conclusion})
    flagged = names(out)
    assert not any("Regurgitation" in n for n in flagged)


def test_genuine_regurgitation_phrases_and_tokens_still_trigger():
    """Spelled-out and bounded abbreviation tokens must still correctly trigger."""
    # 1. Spelled-out tricuspid regurgitation
    out1 = evaluate_v4({"impression_text": "mild tricuspid regurgitation which is centrally directed"})
    assert "Mild Tricuspid Regurgitation" in names(out1)

    # 2. Token TR with word boundary
    out2 = evaluate_v4({"impression_text": "Conclusion: Mild TR noted."})
    assert "Mild Tricuspid Regurgitation" in names(out2)

    # 3. Spelled-out aortic regurgitation
    out3 = evaluate_v4({"impression_text": "Moderate aortic regurgitation seen."})
    assert "Moderate Aortic Regurgitation" in names(out3)

    # 4. Token AR with word boundary
    out4 = evaluate_v4({"impression_text": "Mild AR present."})
    assert "Mild Aortic Regurgitation" in names(out4)

    # 5. Token PR with word boundary
    out5 = evaluate_v4({"impression_text": "Mild PR noted."})
    assert "Mild Pulmonary Regurgitation" in names(out5)


def test_finding_text_normal_valve_suppresses_regurgitation():
    out = evaluate_v4({
        "av_finding": "Normal, trileaflet aortic valve",
        "tv_finding": "Normal tricuspid valve",
        "pv_finding": "Normal pulmonic valve",
        "mv_finding": "Normal mitral valve",
    })
    flagged = names(out)
    assert not any("Regurgitation" in n for n in flagged)
    assert not any("Stenosis" in n for n in flagged)


def test_contains_word_boundary_behaviour():
    assert N.contains_word("Normal, trileaflet aortic valve", "tr") is False
    assert N.contains_word("Normal, trileaflet aortic valve", "ar") is False
    assert N.contains_word("Normal PA pressure", "pr") is False
    assert N.contains_word("Mild TR noted", "tr") is True
    assert N.contains_word("Mild AR.", "ar") is True
    assert N.contains_word("Trivial PR;", "pr") is True



@pytest.mark.parametrize("phrase,expected", [
    ("Myxoma in the left atrium", "Myxoma"),
    ("Constrictive pericarditis", "Constrictive Pericarditis"),
    ("Vegetation on the mitral valve", "Vegetation"),
    ("Bicuspid aortic valve", "Bicuspid Aortic Valve"),
])
def test_conclusion_only_diseases(phrase, expected):
    assert expected in names(evaluate_v4({"impression_text": phrase}))


def test_conclusion_only_diseases_are_never_inferred_from_numbers():
    """These 18 must come from the conclusion or not at all."""
    out = names(evaluate_v4({"ef": "25%", "lvidd": "70 mm", "lvids": "60 mm", "la_diameter": "55 mm"}))
    for never in ("Myxoma", "Constrictive Pericarditis", "Vegetation", "Endocarditis",
                  "Myocarditis", "Bicuspid Aortic Valve", "Restrictive Cardiomyopathy"):
        assert never not in out


# ===========================================================================================
# Priority, risk, schema
# ===========================================================================================
def test_output_is_sorted_by_execution_priority():
    out = evaluate_v4({
        "clots_thrombus": "Thrombus present", "ef": "30%", "la_diameter": "46 mm",
        "pasp": "50 mmHg", "septal_wall_motion": "Akinetic",
    })
    priorities = [d["priority"] for d in out["diseases"]]
    assert priorities == sorted(priorities)
    assert out["diseases"][0]["cardiac_disease_name"] == "Intracardiac Thrombus"


def test_risk_is_high_when_three_or_more_diseases():
    out = evaluate_v4({"ef": "40%", "ivsd": "13 mm", "pasp": "50 mmHg", "la_diameter": "44 mm"})
    assert out["risk_level"] == "High"


def test_schema_covers_all_51_parameters():
    assert len(PARAM_MAP) == 51
    assert len(PRIORITY) in (18, 19)                  # Disease categories + Pediatric Review + Exercise Safety
    assert PRIORITY["Exercise Safety"] < PRIORITY["Risk Score"]


def test_every_prediction_carries_its_evidence():
    """A card with no supporting points would be an unexplainable prediction."""
    out = evaluate_v4({"ef": "30%", "ivsd": "16 mm", "pasp": "60 mmHg", "la_diameter": "50 mm"})
    for d in out["diseases"]:
        assert d["supporting_points"], f"{d['cardiac_disease_name']} has no supporting points"
        assert d["cardiac_disease_name"] and d["category"]


# ===========================================================================================
# Exercise safety -- the nine Tier-1 hard-stop groups
# ===========================================================================================
# A report that clears every one of the nine groups. Each test below copies it and breaks
# exactly one group, so a failure names the group that regressed.
CLEARED = {
    "ef": "62%", "av_peak_gradient": "8 mmHg", "mv_peak_gradient": "4 mmHg",
    "pasp": "22 mmHg", "pericardial_effusion": "No pericardial effusion",
    "clots_thrombus": "Absent", "lvot_peak_gradient": "6 mmHg",
    "wall_motion": "Normal", "rwma": "No", "ivsd": "9 mm",
}

EXERCISE_GROUP_NAMES = (
    "Ejection Fraction", "Aortic Stenosis", "Mitral Stenosis", "Pulmonary Pressure (PASP)",
    "Pericardial Effusion", "Clots / Thrombus", "LVOT Obstruction",
    "Regional Wall Motion Abnormality (Ischemia)", "Septal Thickness (HOCM screening)",
)


def safety(params):
    return evaluate_v4(params)["exercise_safety"]


def cleared_with(**overrides):
    return safety({**CLEARED, **overrides})


def test_exercise_safety_is_a_separate_output_key_from_athlete_screening():
    """Different question, different audience -- the two must never be collapsed."""
    out = evaluate_v4(CLEARED)
    assert out["exercise_safety"]["category"] == "Exercise Safety"
    assert (out["athlete_screening"] is None
            or out["athlete_screening"]["category"] == "Athlete Screening")


def test_empty_report_is_indeterminate_and_says_so_rather_than_clearing():
    """The whole point: no data must never read as a safe result."""
    verdict = safety({})
    assert verdict["cardiac_disease_name"] == "Exercise Safety Indeterminate"
    assert verdict["severity"] == "info"
    assert "NOT a safe" in verdict["recommendation"]
    joined = " ".join(verdict["supporting_points"])
    for group in EXERCISE_GROUP_NAMES:
        assert group in joined


def test_a_fully_measured_normal_report_finds_no_contraindication():
    verdict = safety(CLEARED)
    assert verdict["cardiac_disease_name"] == "No Exercise Contraindication Found"
    assert verdict["severity"] == "normal"
    # The limits of the screen are stated in the verdict itself, not buried elsewhere.
    for phrase in ("does not replace clinical assessment", "ECG", "symptom history",
                   "blood pressure response to exertion"):
        assert phrase in verdict["recommendation"]


def test_one_missing_group_downgrades_a_clear_report_to_indeterminate():
    """Eight cleared groups plus one unknown is not a clearance."""
    partial = {k: v for k, v in CLEARED.items() if k != "clots_thrombus"}
    verdict = safety(partial)
    assert verdict["cardiac_disease_name"] == "Exercise Safety Indeterminate"
    assert "Clots / Thrombus" in " ".join(verdict["supporting_points"])


@pytest.mark.parametrize("field,value", [
    ("ef", "25%"),                                  # 1  EF < 30%
    ("av_peak_gradient", "70 mmHg"),                # 2  severe AS
    ("mv_peak_gradient", "24 mmHg"),                # 3  severe MS
    ("pasp", "72 mmHg"),                            # 4  PASP > 60 mmHg
    ("pericardial_effusion", "Large effusion"),     # 5
    ("clots_thrombus", "LV apical thrombus seen"),  # 6
    ("lvot_peak_gradient", "80 mmHg"),              # 7  LVOTO > 50 mmHg
    ("wall_motion", "Hypokinetic"),                 # 8  active RWMA
])
def test_each_group_can_contraindicate_on_its_own(field, value):
    verdict = cleared_with(**{field: value})
    assert verdict["cardiac_disease_name"] == "Exercise Contraindicated"
    assert verdict["severity"] == "severe"


@pytest.mark.parametrize("field,value", [
    ("ef", "34%"),                                  # 1  EF 30-39%
    ("av_peak_gradient", "45 mmHg"),                # 2  moderate AS
    ("mv_peak_gradient", "14 mmHg"),                # 3  moderate MS
    ("pasp", "48 mmHg"),                            # 4  PASP 40-60 mmHg
    ("pericardial_effusion", "Moderate effusion"),  # 5
    ("lvot_peak_gradient", "40 mmHg"),              # 7  LVOTO 30-50 mmHg
    ("ivsd", "16 mm"),                              # 9  possible HOCM
])
def test_each_group_can_restrict_on_its_own(field, value):
    verdict = cleared_with(**{field: value})
    assert verdict["cardiac_disease_name"] == "Exercise Restricted / Supervised Only"
    assert verdict["severity"] == "moderate"


def test_contraindication_outranks_caution_and_carries_both_sets_of_evidence():
    verdict = cleared_with(ef="24%", ivsd="17 mm")
    assert verdict["cardiac_disease_name"] == "Exercise Contraindicated"
    joined = " ".join(verdict["supporting_points"])
    assert "EF is 24%" in joined and "IVSd is 17 mm" in joined


def test_a_group_falls_back_through_its_chain_rather_than_going_unknown():
    """No AV gradient, but a velocity is enough to grade the aortic stenosis group."""
    partial = {k: v for k, v in CLEARED.items() if k != "av_peak_gradient"}
    verdict = safety({**partial, "av_peak_velocity": "4.4 m/s"})
    assert verdict["cardiac_disease_name"] == "Exercise Contraindicated"
    assert "AV peak velocity is 4.4 m/s" in " ".join(verdict["supporting_points"])


def test_a_directly_measured_value_is_never_overridden_by_a_derived_one():
    """PASP was reported; a tricuspid jet must not re-derive it."""
    joined = " ".join(cleared_with(tv_peak_velocity="4.5 m/s",
                                   ivc="Plethoric")["supporting_points"])
    assert "PASP is 22 mmHg (directly reported)" in joined
    assert "ESTIMATED" not in joined


def test_an_estimated_pasp_is_labelled_as_estimated():
    partial = {k: v for k, v in CLEARED.items() if k != "pasp"}
    verdict = safety({**partial, "tv_peak_velocity": "3.0 m/s", "ivc": "Dilated IVC"})
    joined = " ".join(verdict["supporting_points"])
    assert "PASP ESTIMATED" in joined and "Bernoulli" in joined
    # 4 x 3.0^2 = 36 mmHg + RAP 15 mmHg = 51 mmHg -> the caution band.
    assert verdict["cardiac_disease_name"] == "Exercise Restricted / Supervised Only"


def test_an_unreported_ivc_does_not_become_a_normal_right_atrial_pressure():
    partial = {k: v for k, v in CLEARED.items() if k != "pasp"}
    verdict = safety({**partial, "tv_peak_gradient": "30 mmHg"})
    assert "RAP ASSUMED 15 mmHg" in " ".join(verdict["supporting_points"])
    assert verdict["cardiac_disease_name"] == "Exercise Restricted / Supervised Only"


def test_an_ungraded_effusion_is_a_caution_not_a_clearance():
    verdict = cleared_with(pericardial_effusion="Pericardial effusion present")
    assert verdict["cardiac_disease_name"] == "Exercise Restricted / Supervised Only"
    assert "UNGRADED" in " ".join(verdict["supporting_points"])


def test_a_positively_absent_finding_clears_its_group():
    verdict = cleared_with(clots_thrombus="No thrombus seen", pericardial_effusion="No effusion")
    assert verdict["cardiac_disease_name"] == "No Exercise Contraindication Found"


def test_septal_thickness_cautions_but_never_contraindicates_on_its_own():
    """HOCM screening is a flag for further workup, not a hard stop by itself."""
    assert (cleared_with(ivsd="22 mm")["cardiac_disease_name"]
            == "Exercise Restricted / Supervised Only")
    assert (cleared_with(ivsd="14 mm")["cardiac_disease_name"]
            == "No Exercise Contraindication Found")


def test_rwma_reported_normal_clears_but_rwma_present_stops_exercise():
    assert cleared_with(rwma="Yes")["cardiac_disease_name"] == "Exercise Contraindicated"
    assert (cleared_with(septal_wall_motion="Akinetic")["cardiac_disease_name"]
            == "Exercise Contraindicated")
    assert (cleared_with(rwma="No", wall_motion="Normal")["cardiac_disease_name"]
            == "No Exercise Contraindication Found")


# ===========================================================================================
# Age bands
# ===========================================================================================
@pytest.mark.parametrize("raw,expected", [
    ("28", "young"), ("39", "young"), ("40", "middle"), ("45 / Female", "middle"),
    ("65", "middle"), ("66", "older"), ("73 years", "older"), ("90", "older"),
    (None, None), ("", None), ("not in report", None), ("enter manually", None),
    ("0", None), ("999", None), ("abc", None),
])
def test_age_band_is_parsed_from_the_reports_own_age_field(raw, expected):
    assert R.resolve_age_band(raw) == expected


def test_an_unusable_age_falls_back_to_the_standard_thresholds():
    """Guessing someone into a band would be exactly the imputation G1 forbids."""
    assert R.thresholds_for(None).ef_caution == 40.0
    assert R.thresholds_for(None).ischemia_suspicion is False
    assert safety({**CLEARED, "ef": "42%"})["cardiac_disease_name"] == \
        "No Exercise Contraindication Found"


def test_older_band_cautions_at_milder_ef_and_pasp():
    for field, value in (("ef", "42%"), ("pasp", "37 mmHg")):
        params = {**CLEARED, field: value}
        assert evaluate_v4(params, patient_age="73")["exercise_safety"]["cardiac_disease_name"] \
            == "Exercise Restricted / Supervised Only", field
        # The same numbers are unremarkable in the standard band.
        assert evaluate_v4(params, patient_age="50")["exercise_safety"]["cardiac_disease_name"] \
            == "No Exercise Contraindication Found", field


def test_young_band_screens_lvot_and_septal_thickness_harder():
    """Exercise-related sudden death from HCM is concentrated under 40."""
    for field, value in (("lvot_peak_gradient", "35 mmHg"), ("ivsd", "16 mm")):
        params = {**CLEARED, field: value}
        assert evaluate_v4(params, patient_age="28")["exercise_safety"]["cardiac_disease_name"] \
            == "Exercise Contraindicated", field
        assert evaluate_v4(params, patient_age="50")["exercise_safety"]["cardiac_disease_name"] \
            == "Exercise Restricted / Supervised Only", field
    # 13 mm is a caution under 40 and nothing at all in the other bands.
    assert evaluate_v4({**CLEARED, "ivsd": "13.5 mm"},
                       patient_age="28")["exercise_safety"]["cardiac_disease_name"] \
        == "Exercise Restricted / Supervised Only"
    assert evaluate_v4({**CLEARED, "ivsd": "13.5 mm"},
                       patient_age="50")["exercise_safety"]["cardiac_disease_name"] \
        == "No Exercise Contraindication Found"


def test_middle_band_widens_the_ischemia_proxy():
    params = {"ef": "47%", "lvids": "44 mm"}
    assert "Possible Ischemic Heart Disease / CAD" in names(evaluate_v4(params, patient_age="50"))
    assert "Possible Ischemic Heart Disease / CAD" not in names(evaluate_v4(params, patient_age="28"))


def test_every_shifted_threshold_names_the_band_that_shifted_it():
    """A clinician must be able to see WHICH numbers were applied, not just the verdict."""
    for age, expect in (("28", "Young (< 40 years)"), ("50", "Middle-aged (40-65 years)"),
                        ("73", "Older (> 65 years)"), (None, "Age not recorded")):
        verdict = evaluate_v4(CLEARED, patient_age=age)["exercise_safety"]
        assert f"Age band applied: {expect}" in verdict["supporting_points"]


def test_older_band_defaults_to_lower_intensity_even_when_nothing_is_flagged():
    older = evaluate_v4(CLEARED, patient_age="73")["exercise_safety"]["recommendation"]
    young = evaluate_v4(CLEARED, patient_age="28")["exercise_safety"]["recommendation"]
    assert "low intensity" in older and "progress slowly" in older
    assert "normal baseline" in young
    # The limits of the screen are still stated in both.
    for text in (older, young):
        assert "does not replace clinical assessment" in text


# ===========================================================================================
# Compulsory groups drive the grading
# ===========================================================================================
def test_stenosis_severity_comes_from_the_measured_gradient_not_the_adjective():
    """The compulsory numbers are more exact than whatever word the report printed."""
    out = evaluate_v4({"av_finding": "Mild aortic stenosis", "av_peak_gradient": "70 mmHg"})
    assert "Severe Aortic Stenosis" in names(out)


def test_the_disease_card_and_the_exercise_verdict_grade_one_lesion_identically():
    out = evaluate_v4({**CLEARED, "av_peak_gradient": "70 mmHg"})
    assert "Severe Aortic Stenosis" in names(out)
    assert out["exercise_safety"]["cardiac_disease_name"] == "Exercise Contraindicated"
    assert any("severe aortic stenosis" in p.lower()
               for p in out["exercise_safety"]["supporting_points"])


def test_compulsory_coverage_reports_which_groups_could_be_graded():
    full = evaluate_v4(CLEARED)["compulsory_coverage"]
    assert full["groups_present"] == 9 and full["missing"] == []

    partial = evaluate_v4({k: v for k, v in CLEARED.items() if k != "clots_thrombus"})
    assert partial["compulsory_coverage"]["groups_present"] == 8
    assert partial["compulsory_coverage"]["missing"] == ["Clots / Thrombus"]

    assert evaluate_v4({})["compulsory_coverage"]["groups_present"] == 0


def test_the_two_compulsory_group_lists_are_one_list():
    """The review page mirrors these names; two copies would drift apart."""
    assert tuple(name for name, _ in R.COMPULSORY_GROUPS) == R._EXERCISE_GROUPS
    assert len(R.COMPULSORY_GROUPS) == 9


def test_exercise_safety_is_additive_to_the_existing_prediction_output():
    """Every key that was there before must still be there, in its place."""
    out = evaluate_v4(CLEARED)
    for key in ("engine_version", "diseases", "normal_heart", "athlete_screening", "risk",
                "risk_level", "parameters_available", "parameters_total", "disclaimer"):
        assert key in out
    keys = list(out)
    assert keys.index("exercise_safety") == keys.index("athlete_screening") + 1


# ===========================================================================================
# Negation Detection Tests
# ===========================================================================================
def test_negation_no_ar_suppressed():
    """'No AR' in conclusion must NOT flag Aortic Regurgitation."""
    out = evaluate_v4({"conclusion_text": "No AR. LV systolic function is normal."})
    assert not any("Aortic Regurgitation" in name for name in names(out))


def test_negation_no_lvoto_and_sam_suppressed():
    """'No LVOTO' / 'No Systolic Anterior Motion' / 'No LVOTO.' must NOT flag LVOTO."""
    for text in ("No LVOTO", "No LVOTO.", "No Systolic Anterior Motion", "No LVOTO or SAM"):
        out = evaluate_v4({"conclusion_text": text})
        assert not any("LVOTO" in name or "Outflow Tract Obstruction" in name for name in names(out))


def test_negation_valve_field_no_regurgitation():
    """'Mitral Valve Finding: No Regurgitation' must NOT flag Mitral Regurgitation and
    must not appear as positive supporting evidence."""
    out = evaluate_v4({"mitral_valve_finding": "No Regurgitation"})
    assert not any("Mitral Regurgitation" in name for name in names(out))
    # Ensure it is not attached as evidence to any disease
    for d in out["diseases"]:
        for pt in d.get("supporting_points", []):
            assert "No Regurgitation" not in pt


def test_positive_mild_tr_still_fires():
    """'Mild TR' without negation MUST still be flagged as present."""
    out = evaluate_v4({"tricuspid_valve_finding": "Mild TR"})
    assert "Mild Tricuspid Regurgitation" in names(out)

    out_conc = evaluate_v4({"conclusion_text": "Mild TR noted."})
    assert any("Tricuspid Regurgitation" in name for name in names(out_conc))


def test_positive_mild_mr_still_fires():
    """'Mild MR' without negation MUST still be flagged as present."""
    out = evaluate_v4({"mitral_valve_finding": "Mild MR"})
    assert "Mild Mitral Regurgitation" in names(out)

    out_conc = evaluate_v4({"conclusion_text": "Mild MR noted."})
    assert any("Mitral Regurgitation" in name for name in names(out_conc))


def test_negation_no_as_ar_suppressed():
    """'No AS/AR' must NOT flag either Aortic Stenosis or Aortic Regurgitation."""
    out_conc = evaluate_v4({"conclusion_text": "No AS/AR. Normal LV function."})
    assert not any("Aortic" in name for name in names(out_conc))

    out_field = evaluate_v4({"aortic_valve_finding": "No AS/AR"})
    assert not any("Aortic" in name for name in names(out_field))


def test_negation_mixed_clauses():
    """Negation in one clause must not suppress positive findings in adjacent clauses."""
    # Sentence boundary
    out1 = evaluate_v4({"conclusion_text": "No AR. Mild TR."})
    assert not any("Aortic Regurgitation" in name for name in names(out1))
    assert any("Tricuspid Regurgitation" in name for name in names(out1))

    # Conjunction boundary (but)
    out2 = evaluate_v4({"conclusion_text": "No AR, but mild MR noted."})
    assert not any("Aortic Regurgitation" in name for name in names(out2))
    assert any("Mitral Regurgitation" in name for name in names(out2))

    # Comma before positive severity modifier
    out3 = evaluate_v4({"conclusion_text": "No AS/AR, mild TR noted."})
    assert not any("Aortic" in name for name in names(out3))
    assert any("Tricuspid Regurgitation" in name for name in names(out3))


def test_negation_clinical_phrases():
    """Extended clinical negation phrases must suppress positive diagnosis."""
    assert not any("Pericardium" in d.get("category", "")
                   for d in evaluate_v4({"conclusion_text": "No evidence of pericardial effusion"})["diseases"])
    assert not any("Hypertrophic Cardiomyopathy" in name
                   for name in names(evaluate_v4({"conclusion_text": "Ruled out HCM"})))
    assert not any("CAD" in name or "Ischemic" in name
                   for name in names(evaluate_v4({"conclusion_text": "Negative for CAD"})))
    assert not any("Thrombus" in name
                   for name in names(evaluate_v4({"conclusion_text": "Free of thrombus"})))
    assert not any("Mitral Regurgitation" in name
                   for name in names(evaluate_v4({"conclusion_text": "Without significant MR"})))
    assert not any("LVOTO" in name
                   for name in names(evaluate_v4({"conclusion_text": "LVOTO: absent"})))


def test_ruled_out_findings_tracked_in_output():
    """Explicitly negated findings must be tracked in ruled_out_findings."""
    out = evaluate_v4({
        "mitral_valve_finding": "No Regurgitation",
        "conclusion_text": "No AR. No LVOTO. No pericardial effusion.",
    })
    assert "ruled_out_findings" in out
    ruled_out = out["ruled_out_findings"]
    assert any("Aortic Regurgitation" in item for item in ruled_out)
    assert any("Mitral Regurgitation" in item for item in ruled_out)
    assert any("LVOTO" in item for item in ruled_out)
    assert any("Pericardial Effusion" in item for item in ruled_out)

