"""
Unit tests for context-sensitive fallback Prediction and Severity scores.
Verifies that:
1. Qualitative-only findings (no numeric threshold measurement) produce context-sensitive,
   differentiated Prediction (45-65%) and Severity (23-32% in info band) scores rather than static 55%/28%.
2. Findings with strong supporting context (multiple sources, related abnormal values) receive higher fallback scores.
3. Findings with weak supporting context (single mention, normal related fields) receive lower fallback scores.
4. Fully-measured findings (with numeric measurements) retain exact threshold-excess scoring and are flagged as not fallback.
5. `prediction_is_fallback`, `severity_is_fallback`, and `is_fallback_score` flags correctly identify fallback scores.
"""
import pytest
from app.predictor.rules_v4 import evaluate_v4


def test_qualitative_findings_produce_differentiated_fallback_scores():
    """
    Compare two qualitative findings:
    - Report 1: Isolated Mild Tricuspid Regurgitation (normal PASP 20, normal RA, normal RV).
    - Report 2: Mild Tricuspid Regurgitation WITH elevated PASP (48 mmHg) and enlarged RA.
    """
    rep1 = evaluate_v4({
        "tv_finding": "Mild Tricuspid Regurgitation",
        "pasp": "20 mmHg",
        "ra_size": "Normal",
        "rv_size": "Normal",
    })
    tr1 = next(d for d in rep1["diseases"] if "Tricuspid" in d["cardiac_disease_name"])

    rep2 = evaluate_v4({
        "tv_finding": "Mild Tricuspid Regurgitation",
        "pasp": "48 mmHg",
        "ra_size": "Enlarged",
        "rv_size": "Dilated",
        "conclusion_text": "Tricuspid regurgitation present with pulmonary hypertension",
    })
    tr2 = next(d for d in rep2["diseases"] if "Tricuspid" in d["cardiac_disease_name"])

    # Both are fallback predictions because no direct TR jet measurement was provided
    assert tr1["prediction_is_fallback"] is True
    assert tr2["prediction_is_fallback"] is True
    assert tr1["severity_is_fallback"] is True
    assert tr2["severity_is_fallback"] is True

    # Scores must be context-differentiated (not identical 55% / 28%)
    assert tr1["prediction_score"] != tr2["prediction_score"]
    assert tr1["severity_score"] != tr2["severity_score"]

    # Strong context (rep2) yields higher scores than weak context (rep1)
    assert tr2["prediction_score"] > tr1["prediction_score"]
    assert tr2["severity_score"] > tr1["severity_score"]

    # Bounded ranges verified
    assert 45 <= tr1["prediction_score"] <= 65
    assert 45 <= tr2["prediction_score"] <= 65
    assert 20 <= tr1["severity_score"] <= 35
    assert 20 <= tr2["severity_score"] <= 35


def test_fully_measured_numeric_findings_are_not_fallback():
    """A report with direct numeric measurements (e.g. EF 30%) uses exact threshold excess."""
    rep = evaluate_v4({"ef": "30%", "lvidd": "64 mm", "lvids": "48 mm"}, patient_age="50")
    dcm = next(d for d in rep["diseases"] if "Dilated" in d["cardiac_disease_name"])

    assert dcm["prediction_is_fallback"] is False
    assert dcm["severity_is_fallback"] is False
    assert dcm["is_fallback_score"] is False
    assert dcm["prediction_score"] > 65  # High excess yields > 65%


def test_multiple_valve_findings_in_same_report_produce_differentiated_scores():
    """
    Verify that Aortic, Tricuspid, and Pulmonary Regurgitation in the SAME report
    receive distinct, differentiated fallback scores rather than identical values.
    """
    rep = evaluate_v4({
        "av_finding": "Aortic Regurgitation",
        "tv_finding": "Tricuspid Regurgitation",
        "pv_finding": "Pulmonary Regurgitation",
    })

    diseases_by_name = {d["cardiac_disease_name"]: d for d in rep["diseases"]}
    ar = diseases_by_name.get("Aortic Regurgitation")
    tr = diseases_by_name.get("Tricuspid Regurgitation")
    pr = diseases_by_name.get("Pulmonary Regurgitation")

    assert ar is not None and tr is not None and pr is not None

    pred_scores = {ar["prediction_score"], tr["prediction_score"], pr["prediction_score"]}
    sev_scores = {ar["severity_score"], tr["severity_score"], pr["severity_score"]}

    # All three findings must have distinct prediction and severity scores
    assert len(pred_scores) == 3, f"Expected 3 distinct prediction scores, got {pred_scores}"
    assert len(sev_scores) == 3, f"Expected 3 distinct severity scores, got {sev_scores}"

