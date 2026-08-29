"""
Blank-parameter dropdowns (value_table.py) and the three independent §9 scores.

The two things that would quietly hurt a user here: a dropdown band that resolves to a value the
rule engine then grades into the WRONG severity, and a doctor-selected band scoring as confidently
as a measured one.
"""
import pytest
from fastapi.testclient import TestClient

from app import models
from app.database import SessionLocal, init_db
from app.main import app
from app.predictor import normalize_v4 as N
from app.predictor import value_table as V
from app.predictor.rules_v4 import evaluate_v4


@pytest.fixture(scope="module")
def client():
    init_db()
    return TestClient(app)


@pytest.fixture(scope="module")
def auth(client):
    email = "value-table-tests@example.com"
    resp = client.post("/api/auth/signup",
                       json={"full_name": "Dr Test", "email": email, "password": "secret123"})
    token = (resp.json()["access_token"] if resp.status_code == 201 else
             client.post("/api/auth/login",
                         data={"username": email, "password": "secret123"}).json()["access_token"])
    return {"headers": {"Authorization": f"Bearer {token}"}, "email": email}


def _make_report(email, **params):
    db = SessionLocal()
    doctor = db.query(models.Doctor).filter_by(email=email).one()
    report = models.CardiacReport(doctor_id=doctor.id, patient_name="Test Patient",
                                  status="extracted", **params)
    db.add(report)
    db.commit()
    uid = report.report_uid
    db.close()
    return uid


# ===========================================================================================
# Age groups
# ===========================================================================================
@pytest.mark.parametrize("age,expected", [
    ("3", "children"), ("14", "children"), ("15", "youth"), ("24", "youth"),
    ("25", "adults"), ("64", "adults"), ("65", "elderly"), ("88 years", "elderly"),
])
def test_age_group_bands(age, expected):
    assert V.resolve_age_group(age)[0] == expected


def test_unknown_age_falls_back_to_adults_and_says_so():
    group, unknown = V.resolve_age_group(None)
    assert (group, unknown) == ("adults", True)
    assert V.resolve_age_group("not recorded")[1] is True


def test_youth_uses_the_adult_table_unchanged():
    for field in V.DROPDOWN_OPTIONS:
        assert V.VALUE_TABLE[field]["youth"] == V.VALUE_TABLE[field]["adults"]


# ===========================================================================================
# Resolution
# ===========================================================================================
def test_children_resolve_to_nothing_even_for_normal():
    for option in ("Normal", "Mild", "Severe"):
        out = V.resolve("ef", option, "children")
        assert out["stored_value"] is None
        assert out["requires_custom"] is True
        assert out["pending_pediatric_review"] is True


def test_elderly_bands_differ_from_adult_where_documented():
    assert V.resolve("pasp", "Normal", "adults")["display"] == "<= 35 mmHg"
    assert V.resolve("pasp", "Normal", "elderly")["display"] == "<= 40 mmHg"
    assert V.resolve("ivsd", "Mild", "adults")["display"] == "12-14 mm"
    assert V.resolve("ivsd", "Mild", "elderly")["display"] == "13-14 mm"
    # ...and are identical everywhere else.
    assert V.resolve("ef", "Moderate", "adults")["display"] == \
        V.resolve("ef", "Moderate", "elderly")["display"]


def test_sex_specific_bands_and_conservative_fallback():
    male = V.resolve("lvidd", "Normal", "adults", "M")
    female = V.resolve("lvidd", "Normal", "adults", "F")
    unknown = V.resolve("lvidd", "Normal", "adults", None)
    assert "male" in male["display"] and "female" in female["display"]
    # No recorded sex takes the lower (female) limits and says why.
    assert unknown["value"] == female["value"]
    assert "sex is not recorded" in unknown["note"]


def test_options_without_a_published_band_force_custom_entry():
    # The source table prints "--" for moderate septal thickness; nothing is invented for it.
    assert V.resolve("ivsd", "Moderate", "adults")["requires_custom"] is True
    # "mildly above the sex-specific limit" is a sentence, not a number.
    assert V.resolve("lv_mass", "Mild", "adults")["requires_custom"] is True
    assert V.resolve("ef", "Custom", "adults")["requires_custom"] is True


def test_every_option_of_every_field_resolves_without_error():
    for field, options in V.DROPDOWN_OPTIONS.items():
        for option in options:
            for group in ("youth", "adults", "elderly"):
                out = V.resolve(field, option, group)
                assert out["stored_value"] is not None or out["requires_custom"], \
                    f"{field}/{option}/{group} resolves to neither a value nor a custom prompt"


# ===========================================================================================
# The stored value has to survive normalize_v4 into the RIGHT band
# ===========================================================================================
@pytest.mark.parametrize("field,option,group,reader,check", [
    ("ef", "Severe", "adults", N.to_percent, lambda v: v < 35),
    ("ef", "Normal", "adults", N.to_percent, lambda v: v >= 55),
    ("ef", "Moderate", "adults", N.to_percent, lambda v: 35 <= v < 45),
    ("ivsd", "Severe", "adults", N.to_mm, lambda v: v >= 15),
    ("ivsd", "Normal", "adults", N.to_mm, lambda v: v <= 11),
    ("la_diameter", "Normal", "adults", N.to_mm, lambda v: v <= 40),
    ("pasp", "Severe", "elderly", N.to_gradient_mmhg, lambda v: v > 65),
    ("av_peak_velocity", "Severe", "adults", N.to_velocity_ms, lambda v: v >= 4.0),
    ("mv_area", "Severe", "adults", lambda raw: N.to_ratio(raw.split("cm2")[0]),
     lambda v: v <= 1.0),
    ("relative_wall_thickness", "Increased", "adults", N.to_ratio, lambda v: v > 0.42),
    ("e_a_ratio", "Low (Impaired Relaxation)", "adults", N.to_ratio, lambda v: v < 0.8),
    ("e_a_ratio", "High (Restrictive)", "adults", N.to_ratio, lambda v: v > 2.0),
])
def test_stored_value_reparses_inside_its_own_band(field, option, group, reader, check):
    stored = V.resolve(field, option, group)["stored_value"]
    value = reader(stored)
    assert value is not None, f"{field}/{option} stored as {stored!r} did not parse back"
    assert check(value), f"{field}/{option} stored as {stored!r} parsed to {value}, out of band"


def test_qualitative_options_keep_the_words_the_rules_match_on():
    assert N.is_absent(V.resolve("clots_thrombus", "No Clots", "adults")["stored_value"])
    assert N.contains(V.resolve("clots_thrombus", "Clots Present", "adults")["stored_value"],
                      "present")
    assert N.is_absent(V.resolve("pericardial_effusion", "None/Trace", "adults")["stored_value"])
    assert N.severity_of(
        V.resolve("pericardial_effusion", "Small", "adults")["stored_value"]) == "Mild"
    assert N.contains(V.resolve("ivc", "Dilated/Plethoric", "adults")["stored_value"], "plethoric")
    assert N.keyword(V.resolve("ra_size", "Severely Dilated", "adults")["stored_value"]) == "dilated"
    assert N.is_normal(V.resolve("av_finding", "Normal", "adults")["stored_value"])


@pytest.mark.parametrize("field,option,expect", [
    ("ef", "Severe", "Severe Left Ventricular Dysfunction"),
    ("ef", "Moderate", "Moderate Left Ventricular Dysfunction"),
    ("pasp", "Moderate", "Pulmonary Hypertension (Moderate)"),
    ("clots_thrombus", "Clots Present", "Intracardiac Thrombus"),
])
def test_dropdown_value_drives_the_expected_prediction(field, option, expect):
    stored = V.resolve(field, option, "adults")["stored_value"]
    out = evaluate_v4({field: stored}, patient_age="50")
    assert expect in [d["cardiac_disease_name"] for d in out["diseases"]]


# ===========================================================================================
# §9 scores
# ===========================================================================================
def test_scores_are_independent_and_do_not_sum_to_100():
    out = evaluate_v4({"ef": "28%", "lvidd": "64 mm", "lvids": "48 mm"}, patient_age="50")
    dcm = next(d for d in out["diseases"] if d["cardiac_disease_name"].startswith("Dilated"))
    for key in ("confidence_score", "prediction_score", "severity_score"):
        assert 0 <= dcm[key] <= 100
    assert dcm["confidence_score"] + dcm["prediction_score"] + dcm["severity_score"] != 100


def test_more_independent_sources_scores_higher_confidence():
    single = evaluate_v4({"ef": "30%"}, patient_age="50")["diseases"][0]
    multi = evaluate_v4({"ef": "30%", "lvidd": "64 mm", "lvids": "48 mm"},
                        patient_age="50")["diseases"]
    dcm = next(d for d in multi if d["cardiac_disease_name"].startswith("Dilated"))
    assert dcm["confidence_score"] > single["confidence_score"]


def test_doctor_dropdown_value_is_capped_below_a_measured_one():
    stored = V.resolve("ef", "Severe", "adults")["stored_value"]
    measured = evaluate_v4({"ef": stored}, patient_age="50")["diseases"][0]
    chosen = evaluate_v4({"ef": stored}, patient_age="50",
                         field_sources={"ef": "doctor_dropdown"})["diseases"][0]
    assert chosen["confidence_score"] < measured["confidence_score"]
    assert chosen["confidence_score"] <= 35


def test_prediction_score_uses_the_age_resolved_threshold():
    # PASP 48: moderate on the adult band (>35), only just past the elderly one (>40).
    adult = evaluate_v4({"pasp": "48 mmHg"}, patient_age="50")["diseases"][0]
    elderly = evaluate_v4({"pasp": "48 mmHg"}, patient_age="72")["diseases"][0]
    assert adult["prediction_score"] > elderly["prediction_score"]


def test_severity_score_tracks_the_engines_own_grading():
    mild = evaluate_v4({"ef": "50%"}, patient_age="50")["diseases"][0]
    severe = evaluate_v4({"ef": "25%"}, patient_age="50")["diseases"][0]
    assert 25 <= mild["severity_score"] <= 40
    assert 70 <= severe["severity_score"] <= 100


def test_binary_findings_take_context_sensitive_fallback_score():
    thrombus = evaluate_v4({"clots_thrombus": "Thrombus present"})["diseases"][0]
    assert 45 <= thrombus["prediction_score"] <= 65
    assert thrombus["prediction_is_fallback"] is True


def test_scores_are_not_attached_to_non_disease_outputs():
    out = evaluate_v4({"ef": "60%", "lvidd": "48 mm", "ivsd": "9 mm", "pwd": "9 mm",
                       "la_diameter": "34 mm", "pasp": "20 mmHg"}, patient_age="30")
    for key in ("normal_heart", "athlete_screening", "exercise_safety", "risk"):
        entry = out.get(key)
        if entry:
            assert entry["confidence_score"] is None
            assert entry["prediction_score"] is None
            assert entry["severity_score"] is None


def test_pediatric_report_is_flagged():
    out = evaluate_v4({"ef": "60%"}, patient_age="7")
    assert out["age_group"] == "children"
    assert out["pending_pediatric_review"] is True
    assert "BSA/z-score dependent" in out["pediatric_notice"]

    adult = evaluate_v4({"ef": "60%"}, patient_age="40")
    assert adult["pending_pediatric_review"] is False


# ===========================================================================================
# API wiring
# ===========================================================================================
def test_sections_endpoint_serves_the_dropdown_config(client):
    meta = client.get("/api/reports/meta/sections").json()
    dd = meta["dropdowns"]
    assert dd["options"]["ef"] == ["Normal", "Mild", "Moderate", "Severe", "Custom"]
    assert dd["units"]["av_peak_gradient"] == "mmHg"
    assert dd["value_table"]["ef"]["adults"]["Severe"]["value"].startswith("29")
    assert dd["value_table"]["lvidd"]["adults"]["by_sex"]["male"]["Normal"]["value"]
    assert "mmHg" in dd["custom_notes"]["pasp"]
    # Every rendered field has a vocabulary, so no blank field is left without a dropdown.
    rendered = {f for section in meta["sections"] for f in section["fields"]}
    assert rendered <= set(dd["options"])


def test_saving_a_dropdown_selection_records_its_provenance_and_caps_confidence(client, auth):
    uid = _make_report(auth["email"], patient_age="45")
    stored = V.resolve("ef", "Severe", "adults")["stored_value"]

    saved = client.put(f"/api/reports/{uid}",
                       headers=auth["headers"],
                       json={"parameters": {"ef": stored},
                             "parameter_sources": {"ef": {
                                 "source": "doctor_dropdown", "option": "Severe",
                                 "age_group": "adults", "resolved_display": "< 35 %",
                                 "unit": "%"}}}).json()
    assert saved["parameters"]["ef"] == stored
    assert saved["extraction_meta"]["ef"]["source"] == "doctor_dropdown"
    assert saved["extraction_meta"]["ef"]["dropdown_option"] == "Severe"

    pred = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()
    assert pred["age_group"] == "adults"
    lv = next(d for d in pred["diseases"] if d["cardiac_disease_name"].startswith("Severe Left"))
    assert lv["confidence_score"] <= 35


def test_predict_reports_the_age_group_and_pediatric_flag(client, auth):
    uid = _make_report(auth["email"], patient_age="9", ef="60%")
    pred = client.post(f"/api/reports/{uid}/predict", headers=auth["headers"]).json()
    assert pred["age_group"] == "children"
    assert pred["pending_pediatric_review"] is True
