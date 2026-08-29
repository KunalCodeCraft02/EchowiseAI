# Cardiac OCR & Rule Engine: Current Logic Review

> [!NOTE]
> **TESTING / CODE REVIEW ONLY**  
> This document details the exact, current implementation code across the backend OCR extractor, predictor rule engine, value table, API router, and frontend review page as of August 22, 2026.

---

## 1. Segmental Wall Thickness Handling

### A. Real Code: Segmental Extraction in `app/ocr/extractor.py`

```python
# app/ocr/extractor.py (lines 300-338)
def extract_segmental_measurements(text: str) -> Optional[Dict[str, Any]]:
    """Detect when IVS/PW measurements are reported in segmental format (Basal, Mid, Apical)
    e.g. "IVS – Basal: 14mm; Mid: 20mm; Apical: 22mm" or "PW – Basal: 12mm; Mid: 18mm; Apical: 20mm".
    Returns dict with:
      - 'segments': {'basal': '14 mm', 'mid': '20 mm', 'apical': '22 mm'}
      - 'segment_values_mm': {'basal': 14.0, 'mid': 20.0, 'apical': 22.0, 'max': 22.0}
      - 'basal_formatted': '14 mm'
    or None if fewer than 2 segments are found.
    """
    if not text:
        return None
    raw_found: Dict[str, str] = {}
    mm_found: Dict[str, float] = {}

    for seg_key, pattern in _SEGMENT_PATTERNS.items():
        m = pattern.search(text)
        if m:
            num_str, unit = m.group(1), m.group(2)
            try:
                num = float(num_str)
                unit_norm = (unit or "").lower().strip()
                if unit_norm == "cm" or (not unit_norm and num < 3.5):
                    num_mm = round(num * 10.0, 1)
                else:
                    num_mm = round(num, 1)
                raw_found[seg_key] = f"{num_mm:g} mm"
                mm_found[seg_key] = num_mm
            except ValueError:
                continue

    if len(raw_found) >= 2 and "basal" in raw_found:
        max_val = max(mm_found.values())
        mm_found["max"] = max_val
        return {
            "segments": raw_found,
            "segment_values_mm": mm_found,
            "basal_formatted": raw_found["basal"],
        }
    return None
```

```python
# app/ocr/extractor.py (lines 880-892 inside resolve_value)
    # Segmental measurement detection for wall thickness parameters (IVSd / PWd / IVSs / PWs)
    seg_info = None
    if canon in ("IVSd", "PWd", "IVSs", "PWs"):
        seg_info = extract_segmental_measurements(raw_value_text) or extract_segmental_measurements(snippet)
        if seg_info:
            result_meta["segmental"] = True
            result_meta["segments"] = seg_info["segments"]
            result_meta["segment_values_mm"] = seg_info["segment_values_mm"]
            result_meta[f"{db_field}_basal"] = seg_info["segment_values_mm"].get("basal")
            result_meta[f"{db_field}_mid"] = seg_info["segment_values_mm"].get("mid")
            result_meta[f"{db_field}_apical"] = seg_info["segment_values_mm"].get("apical")
            raw_value_text = seg_info["basal_formatted"]
```

---

### B. Real Code: `Params` Ingestion and Hypertrophy Rule in `app/predictor/rules_v4.py`

```python
# app/predictor/rules_v4.py (lines 236-257 inside Params.__init__)
        self.ivsd = N.to_mm(self.raw["ivsd"])
        self.ivss = N.to_mm(self.raw["ivss"])
        self.pwd = N.to_mm(self.raw["pwd"])
        self.pws = N.to_mm(self.raw["pws"])

        meta = raw.get("extraction_meta") or {}
        ivsd_meta = meta.get("ivsd") or {}
        pwd_meta = meta.get("pwd") or {}
        self.ivsd_max = (
            N.to_mm(raw.get("ivsd_max"))
            or ivsd_meta.get("segment_values_mm", {}).get("max")
            or N.to_mm(raw.get("ivsd_apical"))
            or N.to_mm(raw.get("ivsd_mid"))
            or self.ivsd
        )
        self.pwd_max = (
            N.to_mm(raw.get("pwd_max"))
            or pwd_meta.get("segment_values_mm", {}).get("max")
            or N.to_mm(raw.get("pwd_apical"))
            or N.to_mm(raw.get("pwd_mid"))
            or self.pwd
        )
```

```python
# app/predictor/rules_v4.py (lines 664-697 inside _rule_hypertrophy)
def _rule_hypertrophy(p: Params, t: AgeThresholds = _DEFAULT_THRESHOLDS) -> List[Prediction]:
    out: List[Prediction] = []
    points: List[str] = []
    lvh_fields: List[str] = []
    ivsd_thresh = max(filter(None, [p.ivsd, getattr(p, "ivsd_max", None)])) if (p.ivsd is not None or getattr(p, "ivsd_max", None) is not None) else None
    pwd_thresh = max(filter(None, [p.pwd, getattr(p, "pwd_max", None)])) if (p.pwd is not None or getattr(p, "pwd_max", None) is not None) else None

    if ivsd_thresh is not None and ivsd_thresh > 11:
        points.append(f"IVSd is {ivsd_thresh:g} mm (Threshold: > 11 mm)")
        lvh_fields.append("ivsd")
    if pwd_thresh is not None and pwd_thresh > 11:
        points.append(f"PWd is {pwd_thresh:g} mm (Threshold: > 11 mm)")
        lvh_fields.append("pwd")
    if p.lv_mass_index is not None and p.lv_mass_index > 115:
        points.append(f"LV Mass Index is {p.lv_mass_index:g} g/m2 (Threshold: > 115 g/m2)")
        lvh_fields.append("lv_mass")
    for label, value, fname in (("IVSd", p.raw["ivsd"], "ivsd"), ("PWd", p.raw["pwd"], "pwd")):
        if N.contains(value, "thicken"):
            points.append(f"{label} reported as '{N.clean(value)}'")
            if fname not in lvh_fields:
                lvh_fields.append(fname)
    if p.says("lvh", "left ventricular hypertrophy"):
        points.append(_from_conclusion(p, "LVH", "Left Ventricular Hypertrophy"))

    if points:
        severe = ((ivsd_thresh is not None and ivsd_thresh >= 15) or (pwd_thresh is not None and pwd_thresh >= 15))
        if severe:
            points.append("Wall thickness >= 15 mm indicates severe hypertrophy")
        out.append(Prediction("Severe Left Ventricular Hypertrophy" if severe
                              else "Left Ventricular Hypertrophy (LVH)",
                              "Hypertrophy", "severe" if severe else "moderate", points,
                              "Assess for hypertension, aortic stenosis or infiltrative disease.",
                              fields=lvh_fields))
```

---

### C. Specific Questions Answered

1. **Where is the "main" IVSd/PWd value selected?**
   - In `extractor.py:891`: `raw_value_text = seg_info["basal_formatted"]`.
   - The headline database fields (`cardiac_reports.ivsd` and `cardiac_reports.pwd`) store the **Basal value** (e.g. `"14 mm"`), while the complete dictionary `{"basal": "14 mm", "mid": "20 mm", "apical": "22 mm"}` is stored in `cardiac_reports.extraction_meta["ivsd"]["segments"]`.
2. **Is it hardcoded to always use Basal or is there logic to pick the max?**
   - For display in the main card input, it is hardcoded to **Basal** (`basal_formatted`).
   - For rule evaluation, `extractor.py:331` computes `max_val = max(mm_found.values())` and stores it under `extraction_meta["ivsd"]["segment_values_mm"]["max"]`.
   - In `rules_v4.py:Params`, `self.ivsd_max` pulls this max value (`22.0 mm`).
3. **Which downstream rules read the "main" (Basal) value vs. reading the max segmental value?**
   - **Rules reading the MAX segmental value (`ivsd_thresh = max(p.ivsd, p.ivsd_max)`)**:
     - `_rule_hypertrophy` (LVH threshold `> 11 mm` and Severe LVH `ivsd_thresh >= 15 mm`).
     - `_rule_hypertrophy` (HCM screening cut-off `ivsd_thresh >= 13/15 mm`).
   - **Rules reading only the MAIN (Basal) value (`p.ivsd`, `p.pwd`)**:
     - Concentric LVH Geometry / RWT calculation: `2 * p.pwd / p.lvidd` (uses basal `p.pwd`).
     - `_rule_combined`: Hypertensive Heart Disease (`p.ivsd > 11 mm`), High Risk Structural Heart Disease (`p.ivsd > 14 mm`).
     - `_rule_athlete`: Athlete's Heart screening (`p.ivsd > ivsd_ceiling`).
     - `_rule_athlete_gray_zone`: HCM vs Athlete gray-zone differential (`ivsd` in 13–15mm band).
     - `_rule_normal_heart`: Normal study validation (`p.ivsd <= 11 mm`).

---

## 2. Default/Fallback Value Fill Logic

### A. Real Code: Frontend Auto-Normal Default Fill in `report.html`

```javascript
// app/static/pages/report.html (lines 560-583)
  function initAutoNormalDefaults() {
    if (pediatricPending()) return;
    const bsaVal = (report.parameters && report.parameters.bsa) || report.bsa || "";
    const hasBsa = bsaVal != null && String(bsaVal).trim() !== "";
    // Handle both genuinely empty cards (.is-empty) and cards where OCR stored a bare grade word
    // for a numeric field — those render with .has-dropdown but no resolved numeric value yet.
    document.querySelectorAll(".param-card[data-field]").forEach(card => {
      const field = card.dataset.field;
      if (INDEXED_FIELDS.includes(field) && !hasBsa) return; // Do NOT default Normal when BSA missing
      // Only process: (a) empty cards, or (b) cards with a bare grade stored value
      const isEmptyCard = card.classList.contains("is-empty");
      const inp = card.querySelector("input[data-field]");
      const storedValue = (report.parameters || {})[field];
      const isBareGrade = storedValue != null && isBareDradeValue(field, String(storedValue).trim());
      if (!isEmptyCard && !isBareGrade) return;
      const option = normalDropdownOption(field);
      if (!option) return;
      const entry = dropdownEntry(field, option);
      if (!entry || entry.custom || !entry.value) return;
      const select = card.querySelector("select[data-select]");
      if (select) select.value = option;
      applySelection(field, option);
    });
  }
```

```javascript
// app/static/pages/report.html (lines 636-646 inside applySelection)
    input.hidden = false;
    input.readOnly = true;               // the value is the table's, not free text
    input.value = entry.value;
    // The band and the age group it came from. Where a band is sex-specific the display text
    // already names the sex, so it is not repeated here.
    setHelp(`${option}: ${entry.display} — ${ageGroupMeta().label || "Adults"} (from age-group table)`);
    parameterSources[field] = {
      source: "doctor_dropdown", option: option, age_group: patientAgeGroup(),
      resolved_display: entry.display || null, unit: unit,
    };
```

---

### B. Real Code: Value Table Resolution & Numeric Fabrication in `app/predictor/value_table.py`

```python
# app/predictor/value_table.py (lines 453-457)
    "lvot_peak_gradient": {
        "Normal": _n(None, 30, "< 30 mmHg"),
        "Moderate": _n(30, 50, "30-50 mmHg"),
        "Severe": _n(50, None, "> 50 mmHg"),
    },
```

```python
# app/predictor/value_table.py (lines 861-874)
def representative_value(low: Optional[float], high: Optional[float]) -> Optional[float]:
    """The single number that stands in for a band.

    Closed band -> midpoint. Open band -> 15% outside the stated boundary, so the value lands
    unambiguously inside the band rather than exactly on the cut-off, where a `<`/`<=` difference
    would flip the grade.
    """
    if low is not None and high is not None:
        return _round((low + high) / 2.0)
    if high is not None:
        return _round(high * 0.85)
    if low is not None:
        return _round(low * 1.15)
    return None
```

```python
# app/predictor/value_table.py (lines 945-952 inside resolve)
    low, high = entry.get("low"), entry.get("high")
    value = representative_value(low, high)
    out.update({"low": low, "high": high, "value": value})
    if value is None:
        out["requires_custom"] = True
        return out
    out["stored_value"] = f"{value} {unit} ({option})" if unit else f"{value} ({option})"
    return out
```

---

### C. Real Code: `PUT /api/reports` and `POST /predict` in `app/routers/reports.py`

```python
# app/routers/reports.py (lines 264-282 inside update_report)
    if payload.parameter_sources:
        meta = dict(report.extraction_meta or {})
        for field, source in payload.parameter_sources.items():
            if field not in valid_fields:
                raise HTTPException(status_code=400,
                                    detail=f"Unknown parameter field '{field}'.")
            entry = dict(meta.get(field) or {})
            entry.update({
                "source": source.source,
                "value_type": "qualitative" if source.source == "doctor_dropdown" else "numeric",
                "dropdown_option": source.option,
                "dropdown_age_group": source.age_group,
                "dropdown_display": source.resolved_display,
                "unit": source.unit,
                "final_stored_value": payload.parameters.get(field, getattr(report, field)),
                "entered_at": datetime.utcnow().isoformat(),
            })
            meta[field] = entry
        report.extraction_meta = meta
```

```python
# app/routers/reports.py (lines 370-380 inside predict_disease)
    params = {f: getattr(report, f) for f in models.CardiacReport.PARAM_FIELDS}
    params["impression_text"] = report.impression_text
    result = evaluate_v4(params, patient_age=report.patient_age,
                         patient_sex=report.patient_gender,
                         field_sources=_field_sources(report),
                         is_athlete=bool(report.is_athlete))
```

---

### D. Specific Questions Answered

1. **When a field is filled with a default, is that fact tracked internally?**
   - **Yes**: The frontend sets `parameter_sources[field] = { source: "doctor_dropdown", option: "Normal", ... }`.
   - The backend `PUT /api/reports/{id}` saves this in `extraction_meta[field]["source"] = "doctor_dropdown"`.
   - `_field_sources(report)` extracts `{field: "doctor_dropdown"}` and passes it to `evaluate_v4(..., field_sources=...)`.
2. **Do the disease-prediction rules (rules_v4.py) check this flag before triggering a prediction?**
   - **NO!** The prediction rules (`_rule_lvoto`, `_rule_valves`, `_rule_hypertrophy`, `_rule_ischemia`, etc.) only inspect `p.lvot_grad`, `p.ef`, `p.ivsd`, etc.
   - The `dropdown_fields` set is **only passed to `_apply_scores()`** to cap confidence scores (e.g. capping confidence at 45%–55%) **AFTER the disease prediction has already fired**.
   - Therefore, a fabricated numeric value from a dropdown default can trigger disease rules.
3. **Trace: LVOT Peak Gradient defaulting to 25.5 mmHg causing False Positive LVOTO**:
   1. **Definition**: In `value_table.py:454`, `lvot_peak_gradient` Normal band is defined as `< 30 mmHg` (`high = 30`).
   2. **Fabrication**: `representative_value(None, 30)` computes `30 * 0.85 = 25.5`.
   3. **Auto-Fill**: For a report where LVOT was not measured, `report.html:initAutoNormalDefaults()` writes `"25.5 mmHg (Normal)"` into the LVOT field.
   4. **Save**: The doctor clicks "Save Review", persisting `cardiac_reports.lvot_peak_gradient = "25.5 mmHg (Normal)"`.
   5. **Age Threshold for Young Patients**: In `rules_v4.py:145`, for a patient under 40 years old (`"young"` band), `t.lvot_grad_caution = 20.0 mmHg`.
   6. **Rule Evaluation**: In `rules_v4.py:1084` (`_rule_lvoto`):
      ```python
      if p.lvot_grad is not None and p.lvot_grad > t.lvot_grad_caution:  # 25.5 > 20.0 is True!
          points.append(f"LVOT peak gradient is {p.lvot_grad:g} mmHg (Threshold: > {t.lvot_grad_caution:g} mmHg)")
      ```
   7. **False Positive Fired**: `_rule_lvoto` generates **`Prediction("Left Ventricular Outflow Tract Obstruction (LVOTO)", "Outflow Tract", "severe", ...)`**.

---

## 3. Age-Based Threshold Logic

### A. Real Code: Age Group & Age Band Resolution in `app/predictor/value_table.py` & `app/predictor/rules_v4.py`

```python
# app/predictor/value_table.py (lines 65-102)
AGE_GROUPS: Dict[str, Dict[str, Any]] = {
    CHILDREN: {
        "label": "Children (0-14 years)",
        "min_age": 0, "max_age": 14,
        # The whole point of this flag: the pediatric table is intentionally EMPTY.
        "pending_pediatric_review": True,
        "force_custom_entry": True,
        "flag": "Pediatric reference ranges not yet clinically validated",
        "notice": ("Pediatric norms are BSA/z-score dependent rather than fixed numbers, so no "
                   "dropdown option resolves to a value for this age group — enter the exact "
                   "measured value for every parameter."),
    },
    YOUTH: {
        "label": "Youth (15-24 years)",
        "min_age": 15, "max_age": 24,
        "pending_pediatric_review": False,
        "force_custom_entry": False,
        "uses_table_of": ADULTS,
        "flag": "", "notice": "",
    },
    ADULTS: {
        "label": "Adults (25-64 years)",
        "min_age": 25, "max_age": 64,
        "pending_pediatric_review": False,
        "force_custom_entry": False,
        "flag": "", "notice": "",
    },
    ELDERLY: {
        "label": "Elderly (65+ years)",
        "min_age": 65, "max_age": 120,
        "pending_pediatric_review": False,
        "force_custom_entry": False,
        "flag": "", "notice": "",
    },
}
```

```python
# app/predictor/value_table.py (lines 110-130)
def resolve_age_group(raw_age: Any) -> Tuple[Optional[str], bool]:
    """'8 years' -> ('children', False); None -> ('adults', True)."""
    if raw_age is None:
        return DEFAULT_AGE_GROUP, True
    match = _AGE_RE.search(str(raw_age))
    if not match:
        return DEFAULT_AGE_GROUP, True
    age = int(match.group(1))
    if not 0 <= age <= 120:
        return DEFAULT_AGE_GROUP, True
    if age <= 14:
        return CHILDREN, False
    if age <= 24:
        return YOUTH, False
    if age <= 64:
        return ADULTS, False
    return ELDERLY, False
```

```python
# app/predictor/rules_v4.py (lines 94-114)
def resolve_age_band(raw_age) -> Optional[str]:
    """'73 years' / '73' / '73 / Female' -> 'older'. None when no usable age was recorded."""
    text = N.clean(raw_age)
    if text is None:
        return None
    match = _AGE_RE.search(text)
    if not match:
        return None
    age = int(match.group(1))
    if not 0 < age <= 120:
        return None
    if age < 40:
        return "young"
    if age <= 65:
        return "middle"
    return "older"
```

```python
# app/predictor/rules_v4.py (lines 140-166)
AGE_THRESHOLDS = {
    "young": AgeThresholds(
        label="Young (< 40 years)",
        lvot_grad_contraindicated=30.0, lvot_grad_caution=20.0,
        lvot_vel_contraindicated=2.7, lvot_vel_caution=2.0,
        ivsd_contraindicated=15.0, ivsd_caution=13.0,
        clear_advice="No structural contraindication in a patient under 40; a normal baseline "
                     "exercise tolerance can be assumed and full-intensity training progressed "
                     "as tolerated.",
    ),
    "middle": AgeThresholds(
        label="Middle-aged (40-65 years)",
        ischemia_suspicion=True,
        clear_advice=_STANDARD_ADVICE,
    ),
    "older": AgeThresholds(
        label="Older (> 65 years)",
        ef_caution=45.0, pasp_caution=35.0,
        clear_advice="No structural contraindication, but start at low intensity and progress "
                     "slowly: over-65 patients decompensate at milder values than the standard "
                     "thresholds capture.",
    ),
}
```

---

### B. Specific Questions Answered

1. **Are there separate threshold tables for pediatric patients (infants/children) vs adults?**
   - **For Dropdown Values**: `VALUE_TABLE["children"]` is intentionally an empty dict (`{}`). Pediatric dropdowns do not resolve to numbers and require manual numeric entry (`force_custom_entry: True`).
   - **For Disease Prediction Rules**: **No**. `rules_v4.py` applies hardcoded adult reference thresholds across all 36 rules (e.g. `IVSd > 11 mm`, `LA > 40 mm`, `LVIDd > 58 mm`, `Aortic Root > 40 mm`).
2. **Is pediatric logic wired into the rules engine, or does that UI warning exist without corresponding backend logic?**
   - The warning banner `"Pediatric reference ranges not yet clinically validated"` is triggered when `age <= 14` (`age_group == "children"`).
   - In the backend, `evaluate_v4` passes `"pending_pediatric_review": True` and `"pediatric_notice": ...` to the API output.
   - However, the underlying rule thresholds in `rules_v4.py` remain the adult thresholds.
3. **What threshold is actually used when `patient_age` is `"6 months"` or `"0.5 Y"`?**
   - Regex `_AGE_RE = re.compile(r"(\d{1,3})")`:
     - For `"6 months"`: Regex extracts integer `6`.
       - `resolve_age_group("6 months")` $\to$ `children` (`0 <= 6 <= 14`).
       - `resolve_age_band("6 months")` $\to$ `young` (`6 < 40`).
       - Dropdowns are locked to free-text custom entry (`pediatricPending() == true`).
       - Predictor applies the `young` thresholds (e.g., `lvot_grad_caution = 20.0 mmHg`, `ivsd_caution = 13.0 mm`) and adult baseline thresholds (e.g., `IVSd > 11 mm`, `LA > 40 mm`).
     - For `"0.5 Y"`: Regex extracts integer `0`.
       - `resolve_age_group("0.5 Y")` $\to$ `children` (`0 <= 0 <= 14`).
       - `resolve_age_band("0.5 Y")` $\to$ `None` (because `not 0 < 0 <= 120`).
       - When age band is `None`, predictor applies `_DEFAULT_THRESHOLDS` (standard adult thresholds, e.g. `lvot_grad_caution = 30.0 mmHg`, `ivsd > 11 mm`).
