import pytest
from app.predictor.rules_v4 import evaluate_v4
from app.ocr import preprocessing as pp
from app.ocr import extractor as ex


def _diseases(res):
    return [d["cardiac_disease_name"] for d in res["diseases"]]


# ===========================================================================================
# 1. Cardiomyopathy Phrasing Variants (>= 3 variants each)
# ===========================================================================================
def test_cardiomyopathy_phrasing_variants():
    # Restrictive CM
    for text in ["Restrictive cardiomyopathy", "RCM with diastolic dysfunction", "Infiltrative cardiomyopathy"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Restrictive Cardiomyopathy" in _diseases(res), f"Failed for '{text}'"

    # Ischemic CM
    for text in ["Ischemic cardiomyopathy", "Ischaemic CM with reduced EF", "Severe ICM"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Ischemic Cardiomyopathy" in _diseases(res), f"Failed for '{text}'"

    # Non-Ischemic CM
    for text in ["Non-ischemic cardiomyopathy", "Nonischaemic cardiomyopathy", "NICM"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Non-Ischemic Cardiomyopathy" in _diseases(res), f"Failed for '{text}'"

    # Dilated CM
    for text in ["Dilated cardiomyopathy", "Features of DCM", "Dilated phenotype with poor systolic function"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Dilated Cardiomyopathy (DCM)" in _diseases(res), f"Failed for '{text}'"

    # Hypertrophic CM (HCM / HOCM / Apical / ASH)
    for text in ["Hypertrophic cardiomyopathy", "Apical HCM", "HOCM with obstruction", "Asymmetric septal hypertrophy (ASH)"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Hypertrophic Cardiomyopathy" in _diseases(res), f"Failed for '{text}'"


# ===========================================================================================
# 2. LV Systolic Dysfunction Phrasing Variants
# ===========================================================================================
def test_lv_systolic_dysfunction_phrasing_variants():
    # Severe
    for text in ["Severely depressed LV systolic function.", "Poor LV systolic function", "Severe left ventricular systolic dysfunction"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Severe Left Ventricular Dysfunction" in _diseases(res), f"Failed for '{text}'"

    # Moderate
    for text in ["Moderately depressed LV systolic function.", "Moderate systolic dysfunction", "Moderate LV dysfunction"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Moderate Left Ventricular Dysfunction" in _diseases(res), f"Failed for '{text}'"

    # Mild
    for text in ["Mildly depressed LV systolic function.", "Mild LV systolic dysfunction", "Mild systolic dysfunction"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Mild Left Ventricular Dysfunction" in _diseases(res), f"Failed for '{text}'"


# ===========================================================================================
# 3. Congenital & Structural Findings (>= 3 variants each)
# ===========================================================================================
def test_congenital_and_structural_variants():
    # ASD
    for text in ["Secundum ASD with left to right shunt", "Atrial septal defect (ASD)", "Ostium secundum defect"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Atrial Septal Defect (ASD)" in _diseases(res), f"Failed for '{text}'"

    # VSD
    for text in ["Perimembranous VSD", "Ventricular septal defect", "Muscular VSD"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Ventricular Septal Defect (VSD)" in _diseases(res), f"Failed for '{text}'"

    # Bicuspid Aortic Valve
    for text in ["Bicuspid aortic valve", "Congenital bicuspid AV", "BAV with mild raphe"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Bicuspid Aortic Valve" in _diseases(res), f"Failed for '{text}'"

    # Prosthetic Valve
    for text in ["Prosthetic valve in mitral position", "Normally functioning mechanical valve", "Bioprosthetic valve with good gradients"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Prosthetic Valve" in _diseases(res), f"Failed for '{text}'"


# ===========================================================================================
# 4. Infarction Variants
# ===========================================================================================
def test_infarction_phrasing_variants():
    # Old MI
    for text in ["Old myocardial infarction", "Prior MI in inferior wall", "Healed MI with regional thinning", "Previous myocardial infarction"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Old Myocardial Infarction" in _diseases(res), f"Failed for '{text}'"

    # Recent MI
    for text in ["Recent myocardial infarction", "Acute MI in anterior territory", "STEMI status post PCI", "Recent infarct"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        assert "Recent Myocardial Infarction" in _diseases(res), f"Failed for '{text}'"


# ===========================================================================================
# 5. Inflammatory, Mass, and Vascular Variants
# ===========================================================================================
def test_mass_inflammatory_vascular_variants():
    # Endocarditis & Vegetation
    for text in ["Infective endocarditis", "Valvular vegetation on anterior mitral leaflet", "Endocarditis with vegetation"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        dis = _diseases(res)
        assert "Endocarditis" in dis or "Vegetation" in dis, f"Failed for '{text}'"

    # Myxoma & Cardiac Mass
    for text in ["Left atrial myxoma", "Atrial myxoma attached to fossa ovalis", "Intracardiac mass in right atrium"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        dis = _diseases(res)
        assert "Myxoma" in dis or "Cardiac Mass" in dis, f"Failed for '{text}'"

    # Aortic Coarctation & Aneurysm
    for text in ["Coarctation of aorta", "Ascending aortic aneurysm", "Dilated ascending aorta"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        dis = _diseases(res)
        assert "Coarctation of Aorta" in dis or "Aortic Root Dilation" in dis, f"Failed for '{text}'"

    # Pulmonary Hypertension
    for text in ["Severe pulmonary hypertension", "Pulmonary arterial hypertension (PAH)", "Pulmonary hypertension"]:
        res = evaluate_v4({"impression_text": f"CONCLUSION: {text}."})
        dis = _diseases(res)
        assert any("Pulmonary Hypertension" in d for d in dis), f"Failed for '{text}'"


# ===========================================================================================
# 6. Safety Net: Unmatched Clinical Findings in Conclusion
# ===========================================================================================
def test_unmatched_findings_safety_net():
    # Unclassified clause containing a clinical diagnostic sentinel
    conc = "CONCLUSION: Normal LV size. Anomalous coronary origin with significant stenosis. No LVOTO."
    res = evaluate_v4({"impression_text": conc})
    unmatched = res.get("unmatched_clinical_findings", [])
    # Should catch stenosis clause without flagging negated No LVOTO
    assert any("stenosis" in u["clause"].lower() for u in unmatched)
    assert not any("lvoto" in u["clause"].lower() for u in unmatched)

    # Fully matched report produces empty unmatched list
    conc_matched = "CONCLUSION: Bicuspid aortic valve. Mild aortic regurgitation. No LVOTO."
    res_matched = evaluate_v4({"impression_text": conc_matched})
    unmatched_matched = res_matched.get("unmatched_clinical_findings", [])
    assert len(unmatched_matched) == 0


# ===========================================================================================
# 7. Real Patient Reports Regression Confirmation
# ===========================================================================================
def test_real_reports_regression():
    # Mr. Hanumanth Narute
    hanumanth_struct = pp.extract_structured_digital_pdf("../ocr_testdata/Mr. Hanumanth Narute.pdf")
    h_extracted, h_raw = ex.extract_parameters_structured(hanumanth_struct)
    h_params = {}
    for k, ef in h_extracted.items():
        if ef.value is not None:
            h_params[ef.db_field] = ef.value
            if ef.meta.get("segmental"):
                h_params[f"{ef.db_field}_basal"] = ef.meta.get(f"{ef.db_field}_basal")
                h_params[f"{ef.db_field}_mid"] = ef.meta.get(f"{ef.db_field}_mid")
                h_params[f"{ef.db_field}_apical"] = ef.meta.get(f"{ef.db_field}_apical")

    impr_text = ex.extract_impression_text(h_raw)
    if impr_text:
        h_params["impression_text"] = impr_text

    h_res = evaluate_v4(h_params, patient_age="49 Y", patient_sex="Male")
    h_dis = _diseases(h_res)

    assert any("Hypertrophy" in d for d in h_dis)
    assert "Hypertrophic Cardiomyopathy" in h_dis
    assert any("Diastolic Dysfunction" in d for d in h_dis)
    assert "Mild Mitral Regurgitation" in h_dis
    assert "Mild Tricuspid Regurgitation" in h_dis

    # Mr. Deepak Das
    deepak_struct = pp.extract_structured_digital_pdf("../ocr_testdata/Mr. Deepak Das.pdf")
    d_extracted, d_raw = ex.extract_parameters_structured(deepak_struct)
    assert d_extracted.get("AV_Peak_Gradient") is not None
    assert d_extracted["MV"].value in ("Mild MR", "Mild Regurgitation", "MVA Adequate, Mild MR")
    assert d_extracted["AV"].value in ("No AR", "No Regurgitation", "Normal", "Trileaflet, Degenerative", "Trileaflet, Degenerative, No AS, No AR")
    assert d_extracted["TV"].value in ("Mild TR", "Mild Regurgitation")
