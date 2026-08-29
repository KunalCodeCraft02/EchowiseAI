"""
BSA (Body Surface Area) and Indexed Fields Helper Module.

Implements BSA resolution order:
1. Direct OCR extracted BSA (tagged "from report")
2. Auto-calculated BSA from height (cm) and weight (kg) via DuBois formula:
   BSA = 0.007184 × height^0.725 × weight^0.425 (tagged "calculated from height/weight")
3. Doctor manual entry (tagged "manual entry")
4. Missing (tagged "BSA: — (required)")

Downstream BSA-dependent fields:
- LVIDd / BSA (mm/m²)  [lvidd_indexed_value]
- LA diameter / BSA (cm/m²) [la_diameter_indexed_value]
- LV Mass Index / LVMI (g/m²) [lv_mass]
"""
import math
from typing import Optional, Tuple, Dict, Any


def calculate_dubois_bsa(height_cm: float, weight_kg: float) -> Optional[float]:
    """Calculate BSA using the DuBois formula: BSA = 0.007184 × height^0.725 × weight^0.425."""
    if height_cm <= 0 or weight_kg <= 0:
        return None
    try:
        bsa = 0.007184 * (height_cm ** 0.725) * (weight_kg ** 0.425)
        return round(bsa, 2)
    except (ValueError, OverflowError):
        return None


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("not in report", "not detected", "enter manually", "n/a", "none", "-", "--"):
        return None
    # Extract leading float
    import re
    match = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def resolve_bsa(params: Dict[str, Any], bsa_source_override: Optional[str] = None) -> Tuple[Optional[float], str, str]:
    """
    Resolves BSA value, source key, and display tag.
    Returns: (bsa_val, bsa_source_key, bsa_tag_string)
      bsa_source_key: "from report" | "calculated" | "manual entry" | "missing"
    """
    # 1. Doctor manual entry override if explicitly flagged as manual
    raw_bsa = params.get("bsa")
    bsa_num = _to_float(raw_bsa)
    
    if bsa_source_override in ("manual entry", "manual", "doctor_custom"):
        if bsa_num is not None and bsa_num > 0:
            return bsa_num, "manual entry", f"BSA: {bsa_num:g} m² (manual entry)"

    # 2. Direct OCR extracted BSA from report
    raw_source = params.get("bsa_source")
    if raw_source == "from report" or (bsa_num is not None and bsa_num > 0 and raw_source not in ("calculated", "manual entry")):
        return bsa_num, "from report", f"BSA: {bsa_num:g} m² (from report)"
        
    # 3. Auto-calculate from height & weight via DuBois formula
    height_num = _to_float(params.get("height"))
    weight_num = _to_float(params.get("weight"))
    if height_num is not None and weight_num is not None and height_num > 0 and weight_num > 0:
        calc_bsa = calculate_dubois_bsa(height_num, weight_num)
        if calc_bsa is not None:
            return calc_bsa, "calculated", f"BSA: {calc_bsa:g} m² (calculated from height/weight)"

    # 4. If BSA value is manually present without explicit source
    if bsa_num is not None and bsa_num > 0:
        source_key = raw_source or "manual entry"
        tag = f"BSA: {bsa_num:g} m² ({source_key})"
        return bsa_num, source_key, tag

    # 5. Missing
    return None, "missing", "BSA: — (required)"


def calculate_devereux_lv_mass(ivsd_cm: float, lvidd_cm: float, pwd_cm: float) -> float:
    """Calculate LV Mass using the Devereux formula (ASE convention):
    LV Mass (g) = 0.8 × 1.04 × [(IVSd + LVIDd + PWd)³ - LVIDd³] + 0.6
    """
    cubic_sum = (ivsd_cm + lvidd_cm + pwd_cm) ** 3
    lvid_cubic = lvidd_cm ** 3
    mass = 0.8 * 1.04 * (cubic_sum - lvid_cubic) + 0.6
    return round(mass, 1)


def compute_indexed_fields(params: Dict[str, Any], bsa_val: Optional[float]) -> Dict[str, Optional[str]]:
    """
    Computes the 3 BSA-dependent indexed fields:
    - lvidd_indexed_value (mm/m²)
    - la_diameter_indexed_value (cm/m²)
    - lv_mass (g/m²)

    If bsa_val is missing (None/<=0), returns empty strings for all 3.
    """
    out: Dict[str, Optional[str]] = {
        "lvidd_indexed_value": None,
        "la_diameter_indexed_value": None,
        "lv_mass": None,
    }

    if bsa_val is None or bsa_val <= 0:
        return out

    # --- 1. LVIDd / BSA (mm/m²) ---
    lvidd_num = _to_float(params.get("lvidd"))
    if lvidd_num is not None and lvidd_num > 0:
        # Convert to mm if stored in cm (<= 15.0 cm)
        lvidd_mm = lvidd_num * 10.0 if lvidd_num <= 15.0 else lvidd_num
        idx = round(lvidd_mm / bsa_val, 1)
        out["lvidd_indexed_value"] = f"{idx:g}"

    # --- 2. LA diameter / BSA (cm/m²) ---
    la_num = _to_float(params.get("la_diameter"))
    if la_num is not None and la_num > 0:
        # Convert to cm if stored in mm (> 15.0 mm)
        la_cm = la_num / 10.0 if la_num > 15.0 else la_num
        idx = round(la_cm / bsa_val, 1)
        out["la_diameter_indexed_value"] = f"{idx:g}"

    # --- 3. LV Mass Index / LVMI (g/m²) ---
    lv_mass_num = _to_float(params.get("lv_mass"))
    ivsd_num = _to_float(params.get("ivsd"))
    pwd_num = _to_float(params.get("pwd"))

    if lv_mass_num is not None and lv_mass_num > 0:
        # If absolute LV mass in grams (> 140 g), divide by BSA to get LVMI
        if lv_mass_num > 140.0:
            idx = round(lv_mass_num / bsa_val, 1)
            out["lv_mass"] = f"{idx:g}"
        else:
            # Already indexed LVMI
            out["lv_mass"] = f"{lv_mass_num:g}"
    elif ivsd_num is not None and lvidd_num is not None and pwd_num is not None:
        ivs_cm = ivsd_num / 10.0 if ivsd_num >= 3.0 else ivsd_num
        lvid_cm = lvidd_num / 10.0 if lvidd_num > 15.0 else lvidd_num
        pw_cm = pwd_num / 10.0 if pwd_num >= 3.0 else pwd_num
        abs_mass = calculate_devereux_lv_mass(ivs_cm, lvid_cm, pw_cm)
        idx = round(abs_mass / bsa_val, 1)
        out["lv_mass"] = f"{idx:g}"

    return out
