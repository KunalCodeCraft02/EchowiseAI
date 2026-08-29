import pytest
from app.ocr import extractor as ex
from app.predictor.rules_v4 import evaluate_v4

def test_hanumanth_narute_end_to_end():
    naruto_lines = [
        ("PATIENT NAME : Mr. Hanumanth Narute", 98.0),
        ("AGE / SEX : 49 Y / Male", 98.0),
        ("REF BY : Dr. Self", 95.0),
        ("Height : 165 cm Weight : 68 kg BSA : 1.76 m2", 96.0),
        ("IVS - Basal: 14mm; Mid: 20mm; Apical: 22mm", 95.0),
        ("PW - Basal: 12mm; Mid: 18mm; Apical: 20mm", 95.0),
        ("LVIDd : 4.0 cm", 97.0),
        ("LVIDs : 2.8 cm", 97.0),
        ("EF : 60 %", 98.0),
        ("FS : 30 %", 97.0),
        ("LA : 38 mm", 98.0),
        ("AO : 32 mm", 98.0),
        ("E/A : 1.4", 98.0),
        ("Grade of Regurgitation: Mild MR No AR Mild TR", 96.0),
        ("Mitral Valve : MVA Adequate , Mild MR", 95.0),
        ("Aortic Valve : Normal, Trileaflet.", 96.0),
        ("Pulmonary Valve : Normal.", 95.0),
        ("Tricuspid Valve : Mild TR , No PH (RVSP/TR: 18 mmHg)", 95.0),
        ("Pericardium : No Effusion", 98.0),
        ("Clot/Vegetation : Absent", 98.0),
        ("IMPRESSION:", 99.0),
        ("Severe Asymmetric Septal / Apical Hypertrophic Cardiomyopathy (Apical HCM).", 97.0),
        ("Concentric LV Hypertrophy.", 97.0),
        ("Mild MR, Mild TR.", 96.0),
        ("Grade I Diastolic LV Dysfunction.", 97.0),
        ("Valves Structurally Normal.", 96.0),
        ("No LVOTO, No SAM.", 97.0),
        ("Preserved LV Systolic Function (EF 60%).", 98.0),
    ]

    doc = {
        "lines": naruto_lines,
        "tables": [],
        "table_grid": [],
        "form_fields": [],
        "source_type": "text",
    }

    extracted, _ = ex.extract_parameters_structured(doc)
    raw_text = "\n".join(l for l, _ in naruto_lines)
    impr_text = ex.extract_impression_text(raw_text)

    # Check extracted values
    assert extracted["MV"].value in ("Mild MR", "Mild Regurgitation", "MVA Adequate, Mild MR", "MVA Adequate , Mild MR")
    assert extracted["TV"].value in ("Mild TR", "Mild Regurgitation")
    assert extracted["AV"].value in ("Normal", "No AR", "Normal, Trileaflet")
    assert extracted["PV"].value == "Normal"
    assert extracted["LVIDd"].value == "40 mm"
    assert extracted["LVIDs"].value == "28 mm"
    assert extracted["IVSd"].value == "14 mm"
    assert extracted["PWd"].value == "12 mm"
    assert extracted["PASP"].value == "18 mmHg"
    assert extracted["IVSd"].meta["segmental"] is True
    assert extracted["IVSd"].meta["segments"] == {"basal": "14 mm", "mid": "20 mm", "apical": "22 mm"}

    # Prepare for evaluation
    params_for_eval = {}
    for k, ef in extracted.items():
        if ef.value is not None:
            params_for_eval[ef.db_field] = ef.value
            # Pass segmental fields
            if ef.meta.get("segmental"):
                params_for_eval[f"{ef.db_field}_basal"] = ef.meta.get(f"{ef.db_field}_basal")
                params_for_eval[f"{ef.db_field}_mid"] = ef.meta.get(f"{ef.db_field}_mid")
                params_for_eval[f"{ef.db_field}_apical"] = ef.meta.get(f"{ef.db_field}_apical")

    if impr_text:
        params_for_eval["impression_text"] = impr_text

    eval_res = evaluate_v4(params_for_eval, patient_age="49 Y", patient_sex="Male")
    disease_names = [d["cardiac_disease_name"] for d in eval_res["diseases"]]

    print("Predicted diseases:", disease_names)
    assert any("Hypertrophy" in name for name in disease_names), "Must predict LVH"
    assert any("Hypertrophic Cardiomyopathy" in name for name in disease_names), "Must predict HCM"
    assert "Mild Mitral Regurgitation" in disease_names, "Must predict Mild MR"
    assert "Mild Tricuspid Regurgitation" in disease_names, "Must predict Mild TR"
    assert "Grade I Diastolic Dysfunction (Impaired Relaxation)" in disease_names, "Must predict Grade I Diastolic Dysfunction"

    # Verify negation
    ruled_out = eval_res.get("ruled_out_findings", [])
    assert any("LVOTO" in r for r in ruled_out), "LVOTO must be ruled out"
    assert any("Aortic Regurgitation" in r for r in ruled_out), "Aortic Regurgitation must be ruled out"
