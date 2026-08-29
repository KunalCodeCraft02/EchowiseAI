"""
Unit tests for BSA (Body Surface Area) resolution, DuBois formula, indexed field calculation,
missing BSA fallback, and confirmation validation.
"""
import pytest
from fastapi.testclient import TestClient

from app.ocr.bsa import calculate_dubois_bsa, resolve_bsa, compute_indexed_fields
from app.main import app
from app.database import get_db, SessionLocal
from app import models


def test_dubois_bsa_calculation():
    # Height = 175 cm, Weight = 70 kg
    # 0.007184 * (175 ^ 0.725) * (70 ^ 0.425) ~= 1.85
    bsa = calculate_dubois_bsa(175, 70)
    assert bsa == 1.85

    # Height = 160 cm, Weight = 60 kg
    bsa2 = calculate_dubois_bsa(160, 60)
    assert bsa2 is not None and 1.60 <= bsa2 <= 1.65


def test_bsa_resolution_order():
    # 1. Direct OCR BSA from report
    params_ocr = {"bsa": "2.01", "height": "175", "weight": "70", "bsa_source": "from report"}
    val, source, tag = resolve_bsa(params_ocr)
    assert val == 2.01
    assert source == "from report"
    assert "from report" in tag

    # 2. Auto-calculate from height & weight
    params_calc = {"bsa": None, "height": "175", "weight": "70"}
    val2, source2, tag2 = resolve_bsa(params_calc)
    assert val2 == 1.85
    assert source2 == "calculated"
    assert "calculated from height/weight" in tag2

    # 3. Manual entry override
    params_manual = {"bsa": "1.95", "height": "175", "weight": "70"}
    val3, source3, tag3 = resolve_bsa(params_manual, bsa_source_override="manual entry")
    assert val3 == 1.95
    assert source3 == "manual entry"
    assert "manual entry" in tag3

    # 4. Neither BSA nor height/weight found (missing)
    params_missing = {"bsa": None, "height": None, "weight": None}
    val4, source4, tag4 = resolve_bsa(params_missing)
    assert val4 is None
    assert source4 == "missing"
    assert "required" in tag4


def test_indexed_fields_calculation():
    # When BSA is 2.01 m²
    # LVIDd = 4.5 cm (45 mm) -> 45 / 2.01 = 22.4 mm/m²
    # LA = 3.8 cm -> 3.8 / 2.01 = 1.9 cm/m²
    # LV Mass = 180 g -> 180 / 2.01 = 89.6 g/m²
    params = {
        "lvidd": "4.5",
        "la_diameter": "3.8",
        "lv_mass": "180",
    }
    indexed = compute_indexed_fields(params, 2.01)
    assert indexed["lvidd_indexed_value"] == "22.4"
    assert indexed["la_diameter_indexed_value"] == "1.9"
    assert indexed["lv_mass"] == "89.6"


def test_missing_bsa_leaves_indexed_fields_unpopulated():
    params = {
        "lvidd": "4.5",
        "la_diameter": "3.8",
        "lv_mass": "180",
    }
    indexed = compute_indexed_fields(params, None)
    assert indexed["lvidd_indexed_value"] is None
    assert indexed["la_diameter_indexed_value"] is None
    assert indexed["lv_mass"] is None


@pytest.fixture(scope="module")
def client():
    from app.database import init_db
    init_db()
    return TestClient(app)


@pytest.fixture(scope="module")
def auth(client):
    email = "bsa-tests@example.com"
    resp = client.post("/api/auth/signup",
                       json={"full_name": "Dr BSA Test", "email": email, "password": "secret123"})
    token = (resp.json()["access_token"] if resp.status_code == 201 else
             client.post("/api/auth/login",
                         data={"username": email, "password": "secret123"}).json()["access_token"])
    return {"headers": {"Authorization": f"Bearer {token}"}, "email": email}


def test_api_validation_blocks_confirming_indexed_field_without_bsa(client, auth):
    db = SessionLocal()
    doctor = db.query(models.Doctor).filter_by(email=auth["email"]).one()
    report = models.CardiacReport(
        doctor_id=doctor.id,
        patient_name="BSA Validation Patient",
        status="extracted",
        lvidd="4.5",
        bsa=None,
        bsa_source="missing",
    )
    db.add(report)
    db.commit()
    report_uid = report.report_uid
    db.close()

    # Attempt to confirm lvidd_indexed_value while BSA is missing
    res = client.put(
        f"/api/reports/{report_uid}",
        headers=auth["headers"],
        json={"confirmed_fields": ["lvidd_indexed_value"]}
    )
    assert res.status_code == 400
    assert "Requires manual BSA entry" in res.json()["detail"]

    # Provide BSA manually and confirm
    res2 = client.put(
        f"/api/reports/{report_uid}",
        headers=auth["headers"],
        json={"parameters": {"bsa": "2.01"}, "confirmed_fields": ["lvidd_indexed_value"]}
    )
    assert res2.status_code == 200
    data = res2.json()
    assert data["parameters"]["lvidd_indexed_value"] == "22.4"
    assert "lvidd_indexed_value" in data["doctor_confirmed"]


def test_no_duplicate_patient_params_in_report_sections():
    """Verify that patient_params section is not in REPORT_SECTIONS, ensuring only the single top section renders."""
    from app.ocr.parameter_dict import REPORT_SECTIONS
    section_keys = [s["key"] for s in REPORT_SECTIONS]
    assert "patient_params" not in section_keys
    section_titles = [s["title"] for s in REPORT_SECTIONS]
    assert "BSA & Physical Parameters" not in section_titles


