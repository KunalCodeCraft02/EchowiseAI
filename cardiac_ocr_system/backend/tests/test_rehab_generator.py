"""
Unit tests for the Unified Exercise Safety & Cardiac Rehabilitation generator.
"""
from unittest.mock import MagicMock, patch
import pytest

from app.predictor.rehab_generator import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_DETAILED,
    SYSTEM_PROMPT_GENERAL,
    generate_unified_rehab_plan,
    route_conditions,
)


def test_data_routing_primary_vs_secondary():
    conditions = [
        {"cardiac_disease_name": "Severe Left Ventricular Hypertrophy", "severity": "severe", "supporting_points": ["IVSd 20mm"]},
        {"cardiac_disease_name": "Apical Hypertrophic Cardiomyopathy", "severity": "severe", "supporting_points": ["Apical HCM"]},
        {"cardiac_disease_name": "Aortic Valve Sclerosis", "severity": "moderate", "supporting_points": ["Degenerative"]},
        {"cardiac_disease_name": "Mild Mitral Regurgitation", "severity": "mild", "supporting_points": ["Mild MR"]},
        {"cardiac_disease_name": "Mild Tricuspid Regurgitation", "severity": "mild", "supporting_points": ["Mild TR"]},
        {"cardiac_disease_name": "Grade I Diastolic Dysfunction", "severity": "mild", "supporting_points": ["E/A 0.7"]},
    ]

    primary, secondary = route_conditions(conditions)

    primary_names = [p["name"] for p in primary]
    secondary_names = [s["name"] for s in secondary]

    assert "Severe Left Ventricular Hypertrophy" in primary_names
    assert "Apical Hypertrophic Cardiomyopathy" in primary_names
    assert "Aortic Valve Sclerosis" in primary_names

    assert "Mild Mitral Regurgitation" in secondary_names
    assert "Mild Tricuspid Regurgitation" in secondary_names
    assert "Grade I Diastolic Dysfunction" in secondary_names


def test_fallback_plan_generation_when_groq_offline():
    patient_context = {
        "age": 62,
        "gender": "Male",
        "ef": 60,
        "ivsd_max": 20.0,
        "pwd_max": 14.0,
    }
    conditions = [
        {"cardiac_disease_name": "Severe Left Ventricular Hypertrophy", "severity": "severe"},
        {"cardiac_disease_name": "Mild Mitral Regurgitation", "severity": "mild"},
    ]
    safety_tier = {
        "tier_level": 2,
        "tier_name": "Tier 2: Medically Supervised Exercise Only",
        "heart_rate_ceiling_bpm": 100,
        "borg_rpe_target": "9–11 (Very Light to Light)",
    }

    with patch("app.predictor.rehab_generator.GROQ_API_KEY", ""):
        markdown = generate_unified_rehab_plan(patient_context, conditions, safety_tier)

    assert "### 🏃‍♂️ Unified Cardiac Rehabilitation & Exercise Plan" in markdown
    assert "Primary Safety Ceiling:" in markdown
    assert "Tier 2: Medically Supervised Exercise Only" in markdown
    assert "Governing Echo Findings:" in markdown
    assert "Severe Left Ventricular Hypertrophy" in markdown
    assert "The F.I.T.T. Prescription" in markdown
    assert "* **Frequency:**" in markdown
    assert "* **Intensity:**" in markdown
    assert "* **Time (Duration):**" in markdown
    assert "* **Type (Modality):**" in markdown
    assert "Strict Contraindications" in markdown
    assert "Secondary Clinical Notes & Monitoring" in markdown
    assert "Mild Mitral Regurgitation" in markdown


def test_groq_api_call_structure():
    patient_context = {"age": 55, "ef": 58, "ivsd_max": 13.0}
    conditions = [{"cardiac_disease_name": "Mild LVH", "severity": "mild"}]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "### 🏃‍♂️ Unified Cardiac Rehabilitation & Exercise Plan\n\n**Primary Safety Ceiling:** Tier 3\n..."
                }
            }
        ]
    }

    with patch("app.predictor.rehab_generator.GROQ_API_KEY", "test_key"), \
         patch("httpx.post", return_value=mock_resp) as mock_post:

        result = generate_unified_rehab_plan(patient_context, conditions)

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]

        assert payload["temperature"] == 0.2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == SYSTEM_PROMPT
        assert "Unified Cardiac Rehabilitation" in result


@pytest.mark.parametrize("mode,expected_prompt", [
    ("detailed", SYSTEM_PROMPT_DETAILED),
    ("general", SYSTEM_PROMPT_GENERAL),
    (None, SYSTEM_PROMPT),          # default, unchanged for existing callers
    ("nonsense", SYSTEM_PROMPT),    # unrecognised value falls back to the original clinical prompt
])
def test_mode_selects_the_correct_system_prompt(mode, expected_prompt):
    patient_context = {"age": 55, "ef": 58, "ivsd_max": 13.0}
    conditions = [{"cardiac_disease_name": "Mild LVH", "severity": "mild"}]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "### 🏃‍♂️ placeholder"}}]
    }

    with patch("app.predictor.rehab_generator.GROQ_API_KEY", "test_key"), \
         patch("httpx.post", return_value=mock_resp) as mock_post:

        kwargs = {} if mode is None else {"mode": mode}
        generate_unified_rehab_plan(patient_context, conditions, **kwargs)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["messages"][0]["content"] == expected_prompt


def test_detailed_and_general_prompts_do_not_alter_the_default():
    """Adding the two patient-facing prompts must not touch the original clinical prompt."""
    assert SYSTEM_PROMPT_DETAILED != SYSTEM_PROMPT
    assert SYSTEM_PROMPT_GENERAL != SYSTEM_PROMPT
    assert "F.I.T.T." in SYSTEM_PROMPT
    assert "talk test" in SYSTEM_PROMPT_DETAILED.lower()
    assert "General Heart Health Overview" in SYSTEM_PROMPT_GENERAL
