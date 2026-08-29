import pytest
from app.ocr import preprocessing as pp
from app.ocr import extractor as ex
from app.predictor.rules_v4 import evaluate_v4


def test_doppler_matrix_table_deepak_das():
    """Verify Doppler matrix parsing on real Deepak Das report."""
    path = "../ocr_testdata/Mr. Deepak Das.pdf"
    struct = pp.extract_structured_digital_pdf(path)
    extracted, _ = ex.extract_parameters_structured(struct)

    assert extracted.get("AV_Peak_Gradient") is not None
    assert extracted["AV_Peak_Gradient"].value == "14 mmHg"
    assert extracted["MV"].value in ("Mild MR", "Mild Regurgitation", "MVA Adequate, Mild MR")
    assert extracted["AV"].value in ("No AR", "No Regurgitation", "Normal", "Trileaflet, Degenerative", "Trileaflet, Degenerative, No AS, No AR")
    assert extracted["TV"].value in ("Mild TR", "Mild Regurgitation")


def test_doppler_matrix_table_hanumanth_narute():
    """Verify Doppler matrix parsing on real Hanumanth Narute report."""
    path = "../ocr_testdata/Mr. Hanumanth Narute.pdf"
    struct = pp.extract_structured_digital_pdf(path)
    extracted, _ = ex.extract_parameters_structured(struct)

    # Gradient was '-' in all columns
    assert extracted.get("AV_Peak_Gradient") is None or extracted["AV_Peak_Gradient"].value is None
    assert extracted["MV"].value in ("Mild MR", "Mild Regurgitation", "MVA Adequate, Mild MR")
    assert extracted["AV"].value in ("No AR", "No Regurgitation", "Normal", "Normal, Trileaflet")
    assert extracted["TV"].value in ("Mild TR", "Mild Regurgitation")


def test_doppler_matrix_synthetic_reordered_columns():
    """Verify Doppler table with reordered columns: Aortic, Mitral, Tricuspid."""
    lines = [
        ("                                        Aortic         Mitral    Tricuspid        ", 98.0),
        ("          Peak gradient (mmHg)          28             -         -                ", 98.0),
        ("          Peak velocity (m/s)           2.6            1.1       0.8              ", 98.0),
        ("          Valve area (cm2)              1.0            3.5       -                ", 98.0),
        ("          Grade of Regurgitation        Mild AR        No MR     Mild TR          ", 98.0),
    ]
    doc = {
        "lines": lines,
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }
    extracted, _ = ex.extract_parameters_structured(doc)

    assert extracted.get("AV_Peak_Gradient") is not None
    assert extracted["AV_Peak_Gradient"].value == "28 mmHg"

    assert extracted.get("AV_Peak_Velocity") is not None
    assert extracted["AV_Peak_Velocity"].value == "2.6 m/s"

    assert extracted.get("MV_Peak_Velocity") is not None
    assert extracted["MV_Peak_Velocity"].value == "1.1 m/s"

    assert extracted.get("TV_Peak_Velocity") is not None
    assert extracted["TV_Peak_Velocity"].value == "0.8 m/s"

    assert extracted.get("AV_Area") is not None
    assert extracted["AV_Area"].value == "1 cm2" or extracted["AV_Area"].value == "1.0 cm2"

    assert extracted.get("MV_Area") is not None
    assert extracted["MV_Area"].value == "3.5 cm2"

    assert extracted["AV"].value in ("Mild AR", "Mild Regurgitation")
    assert extracted["MV"].value in ("No MR", "No Regurgitation", "Normal")
    assert extracted["TV"].value in ("Mild TR", "Mild Regurgitation")


def test_doppler_matrix_synthetic_four_columns_structured_grid():
    """Verify 4-column structured table grid (Mitral, Aortic, Tricuspid, Pulmonic)."""
    table_grid = [
        [
            {"text": "Parameter", "confidence": 99.0},
            {"text": "Mitral", "confidence": 99.0},
            {"text": "Aortic", "confidence": 99.0},
            {"text": "Tricuspid", "confidence": 99.0},
            {"text": "Pulmonic", "confidence": 99.0},
        ],
        [
            {"text": "Peak Gradient (mmHg)", "confidence": 99.0},
            {"text": "-", "confidence": 99.0},
            {"text": "45", "confidence": 99.0},
            {"text": "22", "confidence": 99.0},
            {"text": "12", "confidence": 99.0},
        ],
        [
            {"text": "Peak Velocity (m/s)", "confidence": 99.0},
            {"text": "1.2", "confidence": 99.0},
            {"text": "3.4", "confidence": 99.0},
            {"text": "2.4", "confidence": 99.0},
            {"text": "1.7", "confidence": 99.0},
        ],
        [
            {"text": "Grade of Regurgitation", "confidence": 99.0},
            {"text": "Moderate MR", "confidence": 99.0},
            {"text": "No AR", "confidence": 99.0},
            {"text": "Mild TR", "confidence": 99.0},
            {"text": "Normal", "confidence": 99.0},
        ],
    ]
    doc = {
        "lines": [],
        "tables": [],
        "table_grid": table_grid,
        "form_fields": [],
        "source_type": "table_grid",
    }
    extracted, _ = ex.extract_parameters_structured(doc)

    assert extracted.get("AV_Peak_Gradient") is not None
    assert extracted["AV_Peak_Gradient"].value == "45 mmHg"

    assert extracted.get("TV_Peak_Gradient") is not None
    assert extracted["TV_Peak_Gradient"].value == "22 mmHg"

    assert extracted.get("PV_Peak_Gradient") is not None
    assert extracted["PV_Peak_Gradient"].value == "12 mmHg"

    assert extracted.get("AV_Peak_Velocity") is not None
    assert extracted["AV_Peak_Velocity"].value == "3.4 m/s"

    assert extracted["MV"].value in ("Moderate MR", "Moderate Regurgitation")
    assert extracted["AV"].value in ("No AR", "No Regurgitation", "Normal")
    assert extracted["TV"].value in ("Mild TR", "Mild Regurgitation")


def test_diastolic_dysfunction_all_grades_and_phrasings():
    """Verify Grade I, II, III diastolic dysfunction predictions across diverse phrasing styles."""
    grade_1_phrasings = [
        "Grade I Diastolic LV Dysfunction.",
        "Grade 1 Diastolic Dysfunction",
        "Grade one diastolic lv dysfunction",
        "Diastolic dysfunction grade I",
        "Diastolic dysfunction (grade 1)",
        "Diastolic dysfunction - grade I",
        "Grade 1+ Diastolic Dysfunction",
        "Impaired relaxation pattern",
        "Mild diastolic dysfunction",
    ]
    for phrase in grade_1_phrasings:
        preds = evaluate_v4({"impression_text": f"CONCLUSION: {phrase}"})
        diseases = [d["cardiac_disease_name"] for d in preds["diseases"]]
        assert "Grade I Diastolic Dysfunction (Impaired Relaxation)" in diseases, f"Failed for '{phrase}'"

    grade_2_phrasings = [
        "Grade II Diastolic LV Dysfunction.",
        "Grade 2 Diastolic Dysfunction",
        "Grade two diastolic lv dysfunction",
        "Diastolic dysfunction grade II",
        "Diastolic dysfunction (grade 2)",
        "Diastolic dysfunction - grade II",
        "Grade 2+ Diastolic Dysfunction",
        "Pseudonormal filling pattern",
        "Moderate diastolic dysfunction",
    ]
    for phrase in grade_2_phrasings:
        preds = evaluate_v4({"impression_text": f"CONCLUSION: {phrase}"})
        diseases = [d["cardiac_disease_name"] for d in preds["diseases"]]
        assert "Grade II Diastolic Dysfunction (Pseudonormal)" in diseases, f"Failed for '{phrase}'"

    grade_3_phrasings = [
        "Grade III Diastolic LV Dysfunction.",
        "Grade 3 Diastolic Dysfunction",
        "Grade three diastolic lv dysfunction",
        "Diastolic dysfunction grade III",
        "Diastolic dysfunction (grade 3)",
        "Diastolic dysfunction - grade III",
        "Grade 3+ Diastolic Dysfunction",
        "Restrictive filling pattern",
        "Severe diastolic dysfunction",
    ]
    for phrase in grade_3_phrasings:
        preds = evaluate_v4({"impression_text": f"CONCLUSION: {phrase}"})
        diseases = [d["cardiac_disease_name"] for d in preds["diseases"]]
        assert "Grade III Diastolic Dysfunction (Restrictive Filling)" in diseases, f"Failed for '{phrase}'"

    # Negation
    neg_preds = evaluate_v4({"impression_text": "CONCLUSION: No Diastolic Dysfunction. Normal study."})
    diseases = [d["cardiac_disease_name"] for d in neg_preds["diseases"]]
    assert not any("Diastolic" in d for d in diseases)
    assert any("Diastolic Dysfunction" in r for r in neg_preds.get("ruled_out_findings", []))


def test_narrative_lvot_gradient_extraction():
    """Verify narrative LVOT gradient extraction across multiple clinical phrasing variants."""
    phrasings = [
        ("LVOTO with Maximum Gradients 96 mmHg ,  No Significant PH.", "96 mmHg", 96.0),
        ("Acquired HCM , SAM + ,  LVOTO with PSG of 34 mmHg@ rest", "34 mmHg", 34.0),
        ("Peak LVOT gradient: 55 mmHg", "55 mmHg", 55.0),
        ("LVOT peak gradient 42 mmHg", "42 mmHg", 42.0),
        ("LVOT obstruction with PG 96 mmHg", "96 mmHg", 96.0),
        ("Resting LVOT peak gradient 38 mmHg", "38 mmHg", 38.0),
        ("Maximum gradient across LVOT is 75 mmHg", "75 mmHg", 75.0),
    ]
    for phrase, expected_val, expected_num in phrasings:
        doc = {
            "lines": [(phrase, 95.0)],
            "tables": [],
            "table_grid": [],
            "form_fields": [],
            "source_type": "text",
            "full_text": phrase,
        }
        res, _ = ex.extract_parameters_structured(doc)
        lvot_ef = res.get("LVOT_Peak_Gradient")
        assert lvot_ef is not None, f"Failed to extract for: '{phrase}'"
        assert lvot_ef.value == expected_val
        assert lvot_ef.source == "narrative"

        # Verify prediction evaluation
        pred = evaluate_v4({"lvot_peak_gradient": lvot_ef.value})
        disease_names = [d["cardiac_disease_name"] for d in pred["diseases"]]
        assert "Left Ventricular Outflow Tract Obstruction (LVOTO)" in disease_names
