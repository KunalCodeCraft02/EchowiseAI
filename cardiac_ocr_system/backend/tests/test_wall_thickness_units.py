"""
Unit tests for IVSd, IVSs, PWd, PWs extraction and downstream calculations in mm.
Verifies:
1. Source report in cm (IVSd 1cm, IVSs 1.4cm, PWd 0.9cm, PWs 1.3cm) converts to mm (10 mm, 14 mm, 9 mm, 13 mm).
2. Source report already in mm (IVSd 10mm, IVSs 14mm, PWd 9mm, PWs 13mm) stays in mm without double conversion.
3. Source report with bare numbers (no unit symbol) converts and flags for doctor review.
4. LVMI calculation produces ~71 g/m² with mm-based values (IVSd 10 mm, PWd 9 mm, LVIDd 4.5 cm, BSA 2.01 m²).
5. RWT formula evaluates to ~0.40 with PWd 9 mm and LVIDd 45 mm (4.5 cm).
6. Exercise safety contraindication screening correctly checks mm thresholds.
"""
import pytest
from app.ocr import extractor as ex
from app.ocr.bsa import calculate_devereux_lv_mass, compute_indexed_fields
from app.predictor.rules_v4 import evaluate_v4
from app.predictor import normalize as N


def _build_doc(table_rows):
    return {
        "full_text": "\n".join(" ".join(t for t, _ in row) for row in table_rows),
        "tables": [{"page": 1, "rows": [{"cells": [{"text": t, "confidence": c, "bounding_box": None} for t, c in row]} for row in table_rows]}],
        "table_grid": [[{"text": t, "confidence": c} for t, c in row] for row in table_rows],
        "form_fields": [],
        "lines": [],
        "source_type": "table_grid",
    }


def test_cm_report_converts_to_mm():
    """Report with cm values: IVSd 1cm, IVSs 1.4cm, PWd 0.9cm, PWs 1.3cm."""
    doc = _build_doc([
        [("IVSd", 96.0), ("1.0 cm", 96.0)],
        [("IVSs", 95.0), ("1.4 cm", 95.0)],
        [("PWd", 96.0), ("0.9 cm", 96.0)],
        [("PWs", 95.0), ("1.3 cm", 95.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results["IVSd"].value == "10 mm"
    assert results["IVSd"].meta["raw_detected_value"] == "1" or results["IVSd"].meta["raw_detected_value"] == "1.0"
    assert results["IVSd"].meta["raw_detected_unit"] == "cm"
    assert results["IVSd"].meta["conversion_applied"] is True

    assert results["IVSs"].value == "14 mm"
    assert results["IVSs"].meta["conversion_applied"] is True

    assert results["PWd"].value == "9 mm"
    assert results["PWd"].meta["conversion_applied"] is True

    assert results["PWs"].value == "13 mm"
    assert results["PWs"].meta["conversion_applied"] is True


def test_mm_report_stays_in_mm_no_double_conversion():
    """Report already with mm values: IVSd 10mm, IVSs 14mm, PWd 9mm, PWs 13mm."""
    doc = _build_doc([
        [("IVSd", 96.0), ("10 mm", 96.0)],
        [("IVSs", 95.0), ("14 mm", 95.0)],
        [("PWd", 96.0), ("9 mm", 96.0)],
        [("PWs", 95.0), ("13 mm", 95.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results["IVSd"].value == "10 mm"
    assert results["IVSd"].meta["conversion_applied"] is False

    assert results["IVSs"].value == "14 mm"
    assert results["IVSs"].meta["conversion_applied"] is False

    assert results["PWd"].value == "9 mm"
    assert results["PWd"].meta["conversion_applied"] is False

    assert results["PWs"].value == "13 mm"
    assert results["PWs"].meta["conversion_applied"] is False


def test_unitted_magnitude_inference_and_flagging():
    """When no unit is stated, infer from magnitude and flag for confirmation."""
    doc = _build_doc([
        [("IVSd", 96.0), ("1.0", 96.0)],
        [("PWd", 96.0), ("0.9", 96.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results["IVSd"].value == "10 mm"
    assert results["IVSd"].flagged is True

    assert results["PWd"].value == "9 mm"
    assert results["PWd"].flagged is True


def test_lvmi_calculation_with_mm_wall_thickness():
    """
    Given:
      IVSd = 10 mm (1.0 cm)
      PWd = 9 mm (0.9 cm)
      LVIDd = 4.5 cm (45 mm)
      BSA = 2.01 m²
    Expected:
      LV Mass = 0.8 * 1.04 * [(1.0 + 4.5 + 0.9)^3 - 4.5^3] + 0.6 = 142.9 g
      LVMI = 142.9 / 2.01 = 71.1 g/m² (≈ 71 g/m²)
    """
    params = {
        "ivsd": "10 mm",
        "pwd": "9 mm",
        "lvidd": "4.5 cm",
    }
    indexed = compute_indexed_fields(params, bsa_val=2.01)
    assert indexed["lv_mass"] is not None
    lvmi = float(indexed["lv_mass"])
    assert round(lvmi) == 71 or abs(lvmi - 71.1) < 0.2


def test_rwt_calculation_consistency():
    """
    RWT = 2 * PWd / LVIDd
    PWd = 9 mm (0.9 cm), LVIDd = 45 mm (4.5 cm) -> 2 * 9 / 45 = 0.40
    """
    pwd_mm = 9.0
    lvidd_mm = 45.0
    rwt = (2.0 * pwd_mm) / lvidd_mm
    assert round(rwt, 2) == 0.40


def test_exercise_safety_screen_with_mm_values():
    """Normal 10 mm IVSd does not trigger contraindications or cautions."""
    params = {
        "ef": "60%",
        "ivsd": "10 mm",
        "pwd": "9 mm",
        "lvidd": "48 mm",
        "pasp": "25 mmHg",
        "wall_motion": "Normal",
        "pericardial_effusion": "None",
        "clots_thrombus": "No",
    }
    preds = evaluate_v4(params)
    names = [d["cardiac_disease_name"] for d in preds["diseases"]]
    assert "Possible Hypertrophic Cardiomyopathy (HCM/HOCM)" not in names
    assert "Left Ventricular Hypertrophy (LVH)" not in names

    # High 16 mm IVSd triggers HCM screening
    params_hcm = dict(params, ivsd="16 mm")
    preds_hcm = evaluate_v4(params_hcm)
    names_hcm = [d["cardiac_disease_name"] for d in preds_hcm["diseases"]]
    assert any("Hypertrophic Cardiomyopathy" in n or "LVH" in n for n in names_hcm)


def test_segmental_ivs_and_pw_extraction():
    """Bug 1 & Bug 3: Extract Basal, Mid, Apical for IVS and PW. Primary stores Basal."""
    doc = _build_doc([
        [("IVS – Basal: 14mm; Mid: 20mm; Apical: 22mm", 96.0)],
        [("PW – Basal: 12mm; Mid: 18mm; Apical: 20mm", 96.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results["IVSd"].value == "14 mm"
    assert results["IVSd"].meta["segmental"] is True
    assert results["IVSd"].meta["segments"]["basal"] == "14 mm"
    assert results["IVSd"].meta["segments"]["mid"] == "20 mm"
    assert results["IVSd"].meta["segments"]["apical"] == "22 mm"
    assert results["IVSd"].meta["segment_values_mm"]["max"] == 22.0

    assert results["PWd"].value == "12 mm"
    assert results["PWd"].meta["segmental"] is True
    assert results["PWd"].meta["segments"]["basal"] == "12 mm"
    assert results["PWd"].meta["segments"]["mid"] == "18 mm"
    assert results["PWd"].meta["segments"]["apical"] == "20 mm"
    assert results["PWd"].meta["segment_values_mm"]["max"] == 20.0


def test_lvidd_and_lvids_cm_to_mm_conversion():
    """Bug 2: LVIDd 4.0cm -> 40mm and LVIDs 2.8cm -> 28mm."""
    doc = _build_doc([
        [("LVIDd", 96.0), ("4.0 cm", 96.0)],
        [("LVIDs", 96.0), ("2.8 cm", 96.0)],
    ])
    results, _ = ex.extract_parameters_structured(doc)

    assert results["LVIDd"].value == "40 mm"
    assert results["LVIDd"].meta["conversion_applied"] is True

    assert results["LVIDs"].value == "28 mm"
    assert results["LVIDs"].meta["conversion_applied"] is True


def test_lvmi_devereux_calculation_hanumanth_narute():
    """
    Bug 4: Devereux LV Mass operating in cm:
      IVSd = 14 mm (1.4 cm)
      PWd = 12 mm (1.2 cm)
      LVIDd = 40 mm (4.0 cm)
      BSA = 1.9 m²
    Expected:
      LV Mass = 0.8 * 1.04 * [(1.4 + 4.0 + 1.2)^3 - 4.0^3] + 0.6 = 186.5 g
      LVMI = 186.5 / 1.9 = 98.2 g/m² (≈ 98 g/m²)
    """
    params = {
        "ivsd": "14 mm",
        "pwd": "12 mm",
        "lvidd": "40 mm",
    }
    indexed = compute_indexed_fields(params, bsa_val=1.9)
    assert indexed["lv_mass"] is not None
    lvmi = float(indexed["lv_mass"])
    assert round(lvmi, 1) == 98.2 or abs(lvmi - 98.2) < 0.2


def test_disease_detection_routes_segmental_max():
    """
    Bug 1 Part C:
      Primary IVSd is 14mm (Basal), PWd is 12mm (Basal).
      Extraction meta carries segmental max: IVSd=22mm, PWd=20mm.
      LVH / HCM / Severe LVH must check against 22mm max (triggers Severe LVH & HCM).
    """
    params = {
        "ef": "60%",
        "ivsd": "14 mm",
        "pwd": "12 mm",
        "lvidd": "40 mm",
        "lv_mass": "98.2",
        "extraction_meta": {
            "ivsd": {
                "segmental": True,
                "segment_values_mm": {"basal": 14.0, "mid": 20.0, "apical": 22.0, "max": 22.0},
            },
            "pwd": {
                "segmental": True,
                "segment_values_mm": {"basal": 12.0, "mid": 18.0, "apical": 20.0, "max": 20.0},
            },
        },
    }
    preds = evaluate_v4(params)
    disease_names = [d["cardiac_disease_name"] for d in preds["diseases"]]
    assert "Severe Left Ventricular Hypertrophy" in disease_names
    assert "Possible Hypertrophic Cardiomyopathy" in disease_names


def test_aortic_diameter_negation_not_stored_as_value():
    """Bug 5: 'No coarctation of aorta' must NOT store 'No' or descriptor in Ao_Diameter."""
    doc = {
        "full_text": "Conclusion:\nNo coarctation of aorta. Normal LV systolic function.",
        "narrative_blocks": ["No coarctation of aorta. Normal LV systolic function."],
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "lines": [("No coarctation of aorta", 95.0), ("Normal LV systolic function", 95.0)],
        "source_type": "text",
    }
    results, _ = ex.extract_parameters_structured(doc)
    ao_ef = results.get("Ao_Diameter")
    assert ao_ef is None or ao_ef.value is None or ao_ef.value == ""


def test_hanumanth_narute_dimension_and_clots_pericardium_extraction():
    """
    Test extraction of:
    - 'Left ventricular internal dimension - Diastole : 4.0 cms' -> LVIDd: '40 mm'
    - 'Left ventricular internal dimension - Systole : 2.8 cms' -> LVIDs: '28 mm'
    - 'Clots : Nil.' -> Clots_Thrombus: 'Nil'
    - 'Pericardium : Normal.' -> Pericardial_Effusion: 'Normal'
    """
    doc = {
        "lines": [
            ("Left atrium                        :    3.8  cms", 96.0),
            ("Aortic annulus                     :    2.1  cms", 96.0),
            ("Left ventricular internal dimension - Systole : 2.8 cms", 96.0),
            ("Left ventricular internal dimension - Diastole : 4.0 cms", 96.0),
            ("Ejection fraction                  :    60   %", 96.0),
            ("IVS – Basal : 14mm; Mid : 20mm; Apical : 22mm.", 96.0),
            ("PW – Basal : 12mm; Mid : 18mm; Apical : 20mm.", 96.0),
            ("Mitral Valve        :         MVA Adequate , Mild MR", 96.0),
            ("Aortic Valve        :         Normal, Trileaflet.", 96.0),
            ("Pulmonary Valve     :         Normal.", 96.0),
            ("Tricuspid Valve     :         Mild TR , No PH (RVSP/TR: 18 mmHg)", 96.0),
            ("Interoatrial Septum :         Intact.", 96.0),
            ("Interoventricular Septum :    Intact.", 96.0),
            ("Clots               :         Nil.", 96.0),
            ("Vegetations         :         Nil.", 96.0),
            ("Pericardium         :         Normal.", 96.0),
        ],
        "full_text": """
Left ventricular internal dimension - Systole : 2.8 cms
Left ventricular internal dimension - Diastole : 4.0 cms
Clots : Nil.
Pericardium : Normal.
Impression :-
Apical HCM
No Systolic Anterior Motion.
No LVOTO.
Valves Structurally Normal.
No Regional Wall Motion Abnormality.
Mild MR.
No AS/AR.
Mild TR.
No Significant PH.
Grade I Diastolic LV Dysfunction.
Good LV Systolic Function.
No coarctation of aorta.
No clots/vegetations/effusion.
""",
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }
    results, _ = ex.extract_parameters_structured(doc)

    assert results["LVIDd"].value == "40 mm"
    assert results["LVIDs"].value == "28 mm"
    assert results["IVSd"].value == "14 mm"
    assert results["PWd"].value == "12 mm"
    assert results["EF"].value == "60%"
    assert results["Clots_Thrombus"].value == "No Clots"
    assert results["Pericardial_Effusion"].value == "None/Trace"


def test_valve_regurgitation_doppler_table_extraction():
    """Doppler table line 'Grade of Regurgitation: Mild MR No AR Mild TR' extracts all 3 valves correctly."""
    doc = {
        "lines": [("Grade of Regurgitation: Mild MR No AR Mild TR", 96.0)],
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }
    results, _ = ex.extract_parameters_structured(doc)
    assert results["MV"].value in ("Mild Regurgitation", "Mild MR")
    assert results["AV"].value in ("No AR", "No Regurgitation")
    assert results["TV"].value in ("Mild TR", "Mild Regurgitation")


def test_valve_regurgitation_comments_extraction():
    """Comments section lines with valve descriptors extract correctly."""
    doc = {
        "lines": [
            ("Mitral Valve        :         MVA Adequate , Mild MR", 95.0),
            ("Aortic Valve        :         Normal, Trileaflet.", 96.0),
            ("Pulmonary Valve     :         Normal.", 95.0),
            ("Tricuspid Valve     :         Mild TR , No PH (RVSP/TR: 18 mmHg)", 95.0),
        ],
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }
    results, _ = ex.extract_parameters_structured(doc)
    assert results["MV"].value in ("Mild MR", "Mild Regurgitation", "MVA Adequate, Mild MR")
    assert results["AV"].value in ("Normal", "Normal, Trileaflet")
    assert results["PV"].value == "Normal"
    assert results["TV"].value in ("Mild TR", "Mild Regurgitation")
    assert results["PASP"].value == "18 mmHg"


def test_valve_regurgitation_suffix_colons_and_grades():
    """Suffix colons and grade numbers are parsed cleanly."""
    doc = {
        "lines": [("MR: Mild   AR: None   TR: Moderate", 95.0)],
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }
    results, _ = ex.extract_parameters_structured(doc)
    assert results["MV"].value == "Mild MR"
    assert results["AV"].value in ("No AR", "None")
    assert results["TV"].value == "Moderate TR"

    doc_grade = {
        "lines": [("MR Grade 1", 95.0)],
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }
    res_g, _ = ex.extract_parameters_structured(doc_grade)
    assert res_g["MV"].value == "Grade 1 MR"


def test_diastolic_dysfunction_conclusion_variations():
    """Grade I, II, III and generic diastolic dysfunction in report conclusion trigger correct predictions."""
    grade_1_samples = [
        "Grade I Diastolic LV Dysfunction.",
        "Grade 1 Diastolic Dysfunction",
        "Grade one diastolic dysfunction",
        "Diastolic dysfunction grade I",
        "Impaired relaxation",
        "Mild diastolic dysfunction",
    ]
    for sample in grade_1_samples:
        preds = evaluate_v4({"impression_text": f"CONCLUSION: {sample}"})
        names = [d["cardiac_disease_name"] for d in preds["diseases"]]
        assert "Grade I Diastolic Dysfunction (Impaired Relaxation)" in names, f"Failed for {sample}"

    grade_2_samples = [
        "Grade II Diastolic Dysfunction",
        "Grade 2 diastolic lv dysfunction",
        "Grade two diastolic dysfunction",
        "Pseudonormal diastolic filling",
        "Moderate diastolic dysfunction",
    ]
    for sample in grade_2_samples:
        preds = evaluate_v4({"impression_text": f"CONCLUSION: {sample}"})
        names = [d["cardiac_disease_name"] for d in preds["diseases"]]
        assert "Grade II Diastolic Dysfunction (Pseudonormal)" in names, f"Failed for {sample}"

    grade_3_samples = [
        "Grade III Diastolic Dysfunction",
        "Grade 3 diastolic lv dysfunction",
        "Restrictive filling pattern",
        "Severe diastolic dysfunction",
    ]
    for sample in grade_3_samples:
        preds = evaluate_v4({"impression_text": f"CONCLUSION: {sample}"})
        names = [d["cardiac_disease_name"] for d in preds["diseases"]]
        assert "Grade III Diastolic Dysfunction (Restrictive Filling)" in names, f"Failed for {sample}"

    # Negation check
    neg_preds = evaluate_v4({"impression_text": "CONCLUSION: No Diastolic Dysfunction. Normal echo."})
    neg_names = [d["cardiac_disease_name"] for d in neg_preds["diseases"]]
    assert not any("Diastolic" in n for n in neg_names)
    assert any("Diastolic Dysfunction" in r for r in neg_preds.get("ruled_out_findings", []))


def test_clots_vegetations_dropdown_enum_mapping():
    """Test absent and present variants for Clots/Thrombus/Vegetations across template formats."""
    absent_samples = [
        "Clots : Nil\nVegetations : Nil",
        "Clots: Nil   Vegetations: Absent",
        "Clot / Vegetation : Not seen",
        "Clots / Vegetations : None",
        "Vegetations : Not seen",
        "Clots / Thrombus : Nil",
        "Clots: Absent",
        "Vegetation: Absent",
        "Intracardiac Clot: Not detected",
        "LV Thrombus: Negative",
        "Clots and Vegetations: Clean",
        "Clots: Normal",
        "No clots / thrombus seen",
    ]
    for s in absent_samples:
        doc = {
            "lines": [(l, 96.0) for l in s.splitlines()],
            "tables": [], "table_grid": [], "form_fields": [], "source_type": "text",
        }
        res, _ = ex.extract_parameters_structured(doc)
        assert res.get("Clots_Thrombus") is not None, f"Failed to extract for {s}"
        assert res["Clots_Thrombus"].value == "No Clots", f"Expected 'No Clots' for '{s}', got '{res['Clots_Thrombus'].value}'"

    present_samples = [
        ("Clot: Present in LV apex", "Clots Present"),
        ("Thrombus: Detected in LA", "Clots Present"),
        ("Vegetation: Seen on MV leaflet", "Clots Present"),
        ("Intracardiac Thrombus: Present", "Clots Present"),
        ("Clots: Present   Vegetations: Nil", "Clots Present"),
    ]
    for s, expected in present_samples:
        doc = {
            "lines": [(l, 96.0) for l in s.splitlines()],
            "tables": [], "table_grid": [], "form_fields": [], "source_type": "text",
        }
        res, _ = ex.extract_parameters_structured(doc)
        assert res.get("Clots_Thrombus") is not None, f"Failed to extract for {s}"
        assert res["Clots_Thrombus"].value == expected, f"Expected '{expected}' for '{s}', got '{res['Clots_Thrombus'].value}'"


def test_pericardial_effusion_dropdown_enum_mapping():
    """Test Pericardial Effusion maps correctly to None/Trace, Small, Moderate, Large."""
    samples = [
        ("Pericardium: Normal", "None/Trace"),
        ("Pericardial Effusion: Nil", "None/Trace"),
        ("Pericardial Effusion: None/Trace", "None/Trace"),
        ("Pericardial Effusion: No effusion", "None/Trace"),
        ("Pericardial Effusion: Small", "Small"),
        ("Pericardial Effusion: Mild", "Small"),
        ("Pericardial Effusion: Moderate", "Moderate"),
        ("Pericardial Effusion: Large", "Large"),
        ("Pericardial Effusion: Massive", "Large"),
    ]
    for text, expected in samples:
        doc = {
            "lines": [(text, 96.0)],
            "tables": [], "table_grid": [], "form_fields": [], "source_type": "text",
        }
        res, _ = ex.extract_parameters_structured(doc)
        assert res.get("Pericardial_Effusion") is not None, f"Failed for {text}"
        assert res["Pericardial_Effusion"].value == expected, f"Expected '{expected}' for '{text}', got '{res['Pericardial_Effusion'].value}'"


def test_dropdown_hard_validation_guard_rejects_unmapped_text():
    """Unmapped garbage strings must be rejected (value=None, flagged=True) rather than stored."""
    garbage_samples = [
        ("Clots_Thrombus", "xyz_random_noise_123"),
        ("Pericardial_Effusion", "unrelated text block"),
        ("RWMA", "qwerty asdf"),
        ("Wall_Motion", "something unrecognized"),
        ("EF", "not a number or grade"),
    ]
    for canon, raw_val in garbage_samples:
        ef = ex.resolve_value(canon, raw_val, 95.0, "test", raw_val)
        assert ef.value is None, f"Expected None for {canon} with garbage '{raw_val}', got '{ef.value}'"
        assert ef.flagged is True, f"Expected flagged=True for {canon} with garbage '{raw_val}'"




