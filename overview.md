# Cardiac Echo OCR + Prediction Pipeline Architecture & Technical Reference

**Document Version**: 2.2.0  
**Backend System**: Sushrut MVP Cardiac OCR & Decision Support System  
**Engine Version**: Rule Engine v4.0 Enhanced  

---

## 1. Pipeline Architecture

The cardiac echo processing pipeline takes an uploaded 2D Echocardiogram report (PDF, scanned image, or photo) and transforms it into structured clinical parameters, indexed hemodynamics, predictive disease classifications, and tailored exercise rehabilitation plans.

```
[Doctor Upload] (PDF / JPG / PNG)
       │
       ▼
[Stage 0: Ingestion & Document Verification] ──► app/routers/reports.py & app/ocr/document_check.py
       │
       ▼
[Stage 1: Preprocessing & Rasterization]     ──► app/ocr/preprocessing.py
       │
       ▼
[Stage 2: Geometric & Text Extraction]       ──► app/ocr/text_extraction.py (PaddleOCR / Tesseract / pdfplumber)
       │
       ▼
[Stage 3: Semantic Field Parsing]            ──► app/ocr/extractor.py, parameter_dict.py & groq_extractor.py
       │
       ▼
[Stage 4: Indexed Hemodynamics & BSA]        ──► app/ocr/bsa.py (Devereux LV Mass, LVMI, LA/BSA)
       │
       ▼
[Stage 5: Cross-Validation & Impression]     ──► app/ocr/cross_validation.py & extractor.py
       │
       ▼
[Stage 6: Structured Storage & Audit Meta]   ──► app/models.py (CardiacReport) & database.py
       │
       ▼
[Stage 7: Prediction Rule Engine v4.0]       ──► app/predictor/rules_v4.py & normalize_v4.py
       │
       ▼
[Stage 8: AI Care & Exercise Plan]           ──► app/predictor/groq_client.py
       │
       ▼
[Stage 9: Review UI & Live Recalculation]    ──► app/static/pages/report.html & style.css
```

---

### End-to-End Pipeline Stages

| Stage | Responsibility | Primary Files & Functions | Failure / Degradation Behavior |
| :--- | :--- | :--- | :--- |
| **1. Ingestion** | Handles file upload, size checks (max 25MB), auth, and assigns a UUID report record. | [`app/routers/reports.py:upload_report()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/routers/reports.py#L300-L380) | Rejects non-PDF/image formats or oversized uploads immediately. |
| **2. Document Type Check** | Validates whether the document contains genuine echocardiographic content before processing. | [`app/ocr/document_check.py:looks_like_cardiac_report()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/document_check.py#L40-L110) | Flags non-echo documents with `failure_reason = "wrong_file"` to prevent endless rescanning loops. |
| **3. Preprocessing** | Detects digital PDFs vs scanned images; rasterizes PDF pages at 300 DPI; deskews/binarizes for local OCR. | [`app/ocr/preprocessing.py:prepare_inputs()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/preprocessing.py#L30-L75), [`is_digital_pdf()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/preprocessing.py#L110-L135), [`rasterize_pdf_pages()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/preprocessing.py#L140-L170) | Digital PDFs use `pdfplumber` direct text layer extraction; scans fall back to image OCR. |
| **4. Geometric OCR (Stage 1)** | Extracts text boxes with 2D bounding polygons and clusters them into rows/columns to recover table geometry without cloud APIs. | [`app/ocr/text_extraction.py:run_paddle_with_geometry()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/text_extraction.py#L120-L210), [`run_tesseract_with_geometry()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/text_extraction.py#L220-L305), [`merge_documents()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/text_extraction.py#L550-L610) | PaddleOCR runs first; degrades to Tesseract geometry; degrades finally to flat-line OCR. |
| **5. Semantic Field Parsing (Stage 2)** | Maps geometric table cells, form fields, flat lines, and narrative clauses to canonical parameters. Handles unit conversions (`cm` ↔ `mm`), physiological range checks, and segmental extraction. | [`app/ocr/extractor.py:extract_parameters_structured()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/extractor.py#L1485-L1570), [`resolve_value()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/extractor.py#L735-L875), [`app/ocr/parameter_dict.py`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/parameter_dict.py#L48-L920), [`app/ocr/groq_extractor.py`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/groq_extractor.py#L400-L465) | Deterministic dictionary matcher runs offline; Groq text-only model resolves unmapped synonyms/narratives. |
| **6. Indexed Fields & BSA** | Resolves BSA (from report, DuBois height/weight, or manual entry) and computes Devereux LV Mass, LV Mass Index, and indexed chamber diameters. | [`app/ocr/bsa.py:resolve_bsa()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/bsa.py#L45-L75), [`compute_indexed_fields()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/bsa.py#L80-L130), [`calculate_devereux_lv_mass()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/bsa.py#L135-L160) | If BSA is missing, clears indexed parameters and marks them flagged with `"Requires manual BSA entry"`. |
| **7. Impression Extraction** | Extracts the cardiologist's verbatim impression/conclusion prose. | [`app/ocr/extractor.py:extract_impression_text()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/extractor.py#L1219-L1260) | Stored verbatim; if no Impression heading exists, returns `None` and hides the card. |
| **8. Persistence** | Stores patient details, 45+ parameter columns, raw OCR text, confidence scores, flags, and `extraction_meta` in SQLite. | [`app/models.py:CardiacReport`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/models.py#L28-L180), [`app/ocr/pipeline.py:process_report()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/pipeline.py#L271-L388) | Full audit trail written to JSON in `extractions/<report_uid>.json`. |
| **9. Prediction Rule Engine** | Deterministic Clinical Decision Support System running 18 rule categories. Computes 3 independent scores (Confidence, Prediction, Severity) and Exercise Safety verdicts. | [`app/predictor/rules_v4.py:evaluate_v4()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py#L1900-L2050), [`app/predictor/normalize_v4.py`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/normalize_v4.py#L1-L420) | Evaluated strictly on available data. Missing fields are recorded as unassessed; missing never defaults to normal. |
| **10. AI Care & Rehab Plan** | Generates disease-specific exercise and cardiac rehabilitation recommendations. | [`app/predictor/groq_client.py:generate_disease_care_plans()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/groq_client.py#L120-L210) | Blocked completely if exercise is contraindicated or indeterminate. |
| **11. Doctor Review UI** | Renders interactive form cards, segmental toggles, live recalculation of indexed fields on input change, and doctor confirmation checkboxes. | [`app/static/pages/report.html`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/static/pages/report.html), [`app/static/css/style.css`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/static/css/style.css) | Allows full editing, confirmation, and on-the-fly re-prediction. |

---

### Ownership of Default and Fallback Values

| Pipeline Stage | Ownership & Fallback Behavior |
| :--- | :--- |
| **Extraction Phase (`extractor.py`)** | **Does NOT invent defaults**. If a field is not printed in the report or fails OCR confidence thresholds, it is stored as `None` / empty string and added to `flagged_params` with an explanatory `flag_reason` in `extraction_meta`. |
| **Indexed Field Computation (`bsa.py`)** | If BSA is missing, `lvidd_indexed_value`, `la_diameter_indexed_value`, and `lv_mass` are set to empty (`""`) and flagged for manual doctor entry. |
| **Review UI (`report.html`)** | **Owns default visual suggestions**. Function `initAutoNormalDefaults()` pre-populates empty cards with normal dropdown values resolved for patient age group and sex from `value_table.py` (e.g. Adult LVIDd: 50mm, LVIDs: 32.5mm, Ao Diameter: 34mm) to assist the physician. |
| **Prediction Engine (`rules_v4.py`)** | **Strict "Missing Never Becomes Normal" Principle (G1)**. Unfilled parameters are treated as unassessed (`None`). Rules requiring those parameters do not fire, and missing compulsory inputs prevent issuing a "Normal Heart" or "Exercise Cleared" verdict. |

---

## 2. Field Extraction Logic

### Valve Finding & Regurgitation Extraction

Valve regurgitation and stenosis findings are extracted for:
- **Mitral Valve** (`MV` → `mv_finding`)
- **Aortic Valve** (`AV` → `av_finding`)
- **Tricuspid Valve** (`TV` → `tv_finding`)
- **Pulmonary Valve** (`PV` → `pv_finding`)

#### 1. Exact Code & Regex Used

Extraction combines vocabulary terms, boundary-delimited synonym search, and regex phrase normalization.

In [`app/ocr/parameter_dict.py`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/parameter_dict.py#L935-L960):
```python
SEVERITY_TERMS = (
    "mild to moderate", "moderate to severe", "mild-moderate", "moderate-severe",
    "trivial", "trace", "mild", "moderate", "severe", "gross", "significant",
)

LESION_TERMS = (
    "regurgitation", "regurgitant", "stenosis", "sclerosis", "prolapse", "insufficiency",
    "incompetence", "calcification", "thickening", "vegetation", "effusion", "thrombus",
    "hypokinesia", "akinesia", "dyskinesia", "hypertrophy",
)

DESCRIPTOR_TERMS = (
    "normal", "abnormal", "enlarged", "dilated", "not dilated", "not enlarged",
    "hypertrophied", "thickened", "unremarkable", "wnl", "within normal limits",
    "impaired", "preserved", "reduced", "sclerotic", "calcified", "absent", "present",
    "nil", "none", "intact", "adequate", "good", "poor", "mildly dilated", "moderately dilated",
    "severely dilated", "grossly dilated", "borderline",
)

QUALITATIVE_TERMS = tuple(sorted(
    set(SEVERITY_TERMS) | set(LESION_TERMS) | set(DESCRIPTOR_TERMS),
    key=len, reverse=True,
))
```

In [`app/ocr/extractor.py:finding_phrase()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/extractor.py#L1305-L1345):
```python
_NEGATION_RE = re.compile(r"\b(no|nil|absent|without|free of|negative for)\b", re.IGNORECASE)

def finding_phrase(clause: str, canon: Optional[str] = None) -> Optional[str]:
    low = (clause or "").lower()
    if not low.strip():
        return None

    # Canon-specific lesion terms prevent cross-contamination in compound sentences
    canon_lesions = None
    if canon == "Clots_Thrombus":
        canon_lesions = ("thrombus", "clot", "clots", "thrombi")
    elif canon == "Pericardial_Effusion":
        canon_lesions = ("effusion", "fluid", "tamponade")
    elif canon in ("MV", "AV", "TV", "PV"):
        canon_lesions = ("regurgitation", "regurgitant", "stenosis", "sclerosis", "prolapse",
                         "insufficiency", "incompetence", "calcification", "thickening", "bicuspid")

    severity = _first_term(low, SEVERITY_TERMS)
    lesion = _first_term(low, canon_lesions if canon_lesions is not None else LESION_TERMS)
    descriptor = _first_term(low, DESCRIPTOR_TERMS)

    # Negation handling: "No MR" -> "No Regurgitation"
    if lesion and _NEGATION_RE.search(low):
        return f"No {lesion.title()}"

    if severity and lesion:
        return f"{severity.title()} {lesion.title()}"
    if lesion:
        return lesion.title()
    if descriptor:
        adverb = re.search(r"\b(\w+ly)\s+" + re.escape(descriptor) + r"\b", low)
        if adverb:
            return f"{adverb.group(1).title()} {descriptor.title()}"
        if severity:
            return f"{severity.title()} {descriptor.title()}"
        return descriptor.title()
    return None
```

---

#### 2. Sections Scanned per Field

The parser uses a multi-tier matching strategy across different document sections:

1. **Table Grid / Digital Tables (`table_grid`, `tables`)**:
   - Scanned row-by-row.
   - Searches for valve synonyms (`"Mitral Valve"`, `"MV"`, `"MR"`, `"Aortic Valve"`, `"AV"`, `"AR"`, `"Tricuspid Valve"`, `"TV"`, `"TR"`, `"Pulmonary Valve"`, `"PV"`, `"PR"`).
   - Scans sections labeled `DOPPLER MEASUREMENTS`, `VALVE MEASUREMENTS`, or tabular summaries.
2. **Flat Lines (`lines`)**:
   - Scans all lines for label:value patterns (e.g. `Mitral Valve : MVA Adequate , Mild MR`, `Clots : Nil.`, `Pericardium : Normal.`).
   - Uses `_find_all_label_matches()` and `_extract_text_value()`.
3. **Narrative Prose (`narrative_blocks`, `full_text`)**:
   - Scans sections located by `_locate_narrative_sections()`: headings matching `CONCLUSION`, `IMPRESSION`, `FINDINGS`, `VALVE(S)`, `SUMMARY`, `INTERPRETATION`, `COMMENTS`.
   - Clauses are split on `[.;\n]+` and evaluated with `finding_phrase(clause, canon)`.

---

#### 3. Hybrid Parsing & LLM Prompts

Extraction is **hybrid**:
1. **Deterministic Offline Engine**: Runs first using local geometry, dictionary synonyms, and regex. Free, reproducible, and instantaneous.
2. **Groq Text-Only Semantic Layer**: Called when unmapped label variants exist or narrative blocks need AI understanding.

In [`app/ocr/groq_extractor.py`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/ocr/groq_extractor.py#L271-L313):

**Label Mapping Prompt (`_LABEL_PROMPT`)**:
```text
You map echocardiogram report labels onto a fixed list of canonical field keys.

CANONICAL KEYS (the ONLY values you may return):
{keys}

LABELS TO MAP:
{labels}

Rules -- follow exactly:
1. Return a canonical key only when the label means THE SAME measurement. "LAd (AP)" is the
   left atrial AP diameter, "EF (Simpson)" is the ejection fraction.
2. Return null when the label is not one of these measurements, or you are not sure. A null is
   always better than a wrong mapping -- this is clinical data.
3. A label carrying an indexed suffix ("/BSA", "index", "indexed", "(i)") is NEVER the base
   measurement. Map it to the matching *_Indexed key if one exists, otherwise null.
   "LVEDD/BSA" is NOT "LVIDd".
4. Do not invent keys. Do not return a key that is not in the list above.

Return ONLY a JSON array, no prose and no markdown fence:
[{"label": "<label exactly as given>", "key": "<canonical key or null>"}, ...]
```

**Narrative Findings Prompt (`_NARRATIVE_PROMPT`)**:
```text
You read qualitative findings out of the narrative sections of an
echocardiogram report (Conclusion / Impression / Findings / Valves / Summary / Interpretation).

CANONICAL KEYS (the ONLY values you may return):
{keys}

REPORT TEXT:
{blocks}

Rules -- follow exactly:
1. Extract descriptive findings written as prose, e.g. "Mild mitral regurgitation" -> key "MV",
   value "Mild Regurgitation". Keep the severity grader with the finding.
2. `value` must be a short descriptive phrase copied or lightly normalized from the text
   (e.g. "Normal", "Mild Regurgitation", "Moderately Dilated"). Never invent a number.
3. If the text states a measurement with a number, you may return it, with value_type "numeric".
   Otherwise value_type is "qualitative".
4. Report only what the text actually says. Omit anything you are unsure about.
5. Do not invent keys. Do not return a key that is not in the list above.

Return ONLY a JSON array, no prose and no markdown fence:
[{"key": "...", "value": "...", "value_type": "numeric|qualitative", "evidence": "<the sentence>"}, ...]
```

---

#### 4. Execution Trace Scenario

**Input Report Text**:
```text
DOPPLER MEASUREMENTS :-
                              Mitral         Aortic    Tricuspid
Grade of Regurgitation     Mild MR        No AR       Mild TR

COMMENTS :-
Mitral Valve        :         MVA Adequate , Mild MR
Tricuspid Valve     :         Mild TR , No PH (RVSP/TR: 18 mmHg)
```

**Step-by-Step Parser Execution**:

1. **Table Grid Processing (`match_table_grid`)**:
   - In Doppler table row `Grade of Regurgitation`:
     - Columns map to Mitral (`MV`), Aortic (`AV`), Tricuspid (`TV`).
     - Mitral cell holds `"Mild MR"`. Classified as qualitative → `MV = "Mild MR"` (source: `"table"`).
     - Aortic cell holds `"No AR"`. Classified as qualitative → `AV = "No AR"` (source: `"table"`).
     - Tricuspid cell holds `"Mild TR"`. Classified as qualitative → `TV = "Mild TR"` (source: `"table"`).
2. **Flat Line Processing (`match_flat_lines`)**:
   - Line `Mitral Valve : MVA Adequate , Mild MR`:
     - Label match: `"Mitral Valve"` → `MV`.
     - Value extracted: `"MVA Adequate , Mild MR"`.
   - Line `Tricuspid Valve : Mild TR , No PH (RVSP/TR: 18 mmHg)`:
     - Label match: `"Tricuspid Valve"` → `TV`.
     - Value extracted: `"Mild TR , No PH (RVSP/TR: 18 mmHg)"`.
     - Also matches `PASP` (RVSP/TR: 18 mmHg) → `PASP = "18 mmHg"`.
3. **Narrative Clause Processing (`_clause_narrative_findings`)**:
   - Clause `"MVA Adequate , Mild MR"`: `finding_phrase(clause, "MV")` → `"Mild Regurgitation"`.
   - Clause `"Mild TR , No PH"`: `finding_phrase(clause, "TV")` → `"Mild Regurgitation"`.
4. **Source Priority Resolution (`extract_parameters_structured`)**:
   - Table grid results (`"Mild MR"`, `"Mild TR"`) are non-weak structured hits, outranking flat lines and narrative clauses.
   - Result:
     - **`mv_finding`** = `"Mild MR"` (or `"Mild Regurgitation"`, tagged qualitative, confidence 95–100%)
     - **`tv_finding`** = `"Mild TR"` (or `"Mild Regurgitation"`, tagged qualitative, confidence 95–100%)
5. **Prediction Rule Engine (`rules_v4.py: _rule_valves`)**:
   - Evaluates `p.mv_finding`:
     - `N.contains(finding, "regurgitation", "regurgitant", "insufficiency")` or `N.contains_word(finding, "mr")` is **`True`**.
     - `N.severity_of("Mild MR")` extracts **`"Mild"`**.
     - Emits **`Prediction("Mild Mitral Regurgitation", "Valves", "mild", ["Mitral valve finding: 'Mild MR'"])`**.
   - Evaluates `p.tv_finding`:
     - `N.contains_word(finding, "tr")` is **`True`**.
     - `N.severity_of("Mild TR")` extracts **`"Mild"`**.
     - Emits **`Prediction("Mild Tricuspid Regurgitation", "Valves", "mild", ["Tricuspid valve finding: 'Mild TR'"])`**.

---

## 3. Structured Data Schema

The database model [`CardiacReport`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/models.py#L28-L180) stores parameters as flexible `Text` columns to accommodate both numerical measurements and clinical text findings.

### Complete Parameter Schema

| Field Name (`db_field`) | Canonical Name | Data Type | Canonical Unit | Valid Range | Impossible Range | Default Stored Value | Behavior when Missing vs Ambiguous |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ef` | `EF` | Numeric / Text | `%` | 10 – 80 % | 5 – 95 % | `None` | Missing: `None` (unassessed). Ambiguous: Soft-flagged with range warning. |
| `lvidd` | `LVIDd` | Numeric / Text | `mm` | 20.0 – 70.0 mm | 10.0 – 100.0 mm | `None` | Missing: `None`. If in cm (e.g. 4.0 cm), converted to 40 mm. |
| `lvids` | `LVIDs` | Numeric / Text | `mm` | 10.0 – 60.0 mm | 5.0 – 90.0 mm | `None` | Missing: `None`. If in cm (e.g. 2.8 cm), converted to 28 mm. |
| `ivsd` | `IVSd` | Numeric / Text | `mm` | 5.0 – 25.0 mm | 2.0 – 40.0 mm | `None` | Missing: `None`. Segmental lines store Basal in primary; segments in `meta`. |
| `ivss` | `IVSs` | Numeric / Text | `mm` | 7.0 – 30.0 mm | 3.0 – 45.0 mm | `None` | Missing: `None`. Converted from cm if needed. |
| `pwd` | `PWd` | Numeric / Text | `mm` | 5.0 – 25.0 mm | 2.0 – 40.0 mm | `None` | Missing: `None`. Segmental lines store Basal in primary; segments in `meta`. |
| `pws` | `PWs` | Numeric / Text | `mm` | 7.0 – 30.0 mm | 3.0 – 45.0 mm | `None` | Missing: `None`. Converted from cm if needed. |
| `la_diameter` | `LA_Diameter` | Numeric / Text | `cm` | 1.5 – 6.5 cm | 0.5 – 9.0 cm | `None` | Missing: `None`. Stored in canonical cm. |
| `ao_diameter` | `Ao_Diameter` | Numeric / Text | `mm` | 15.0 – 55.0 mm | 10.0 – 80.0 mm | `None` | Missing: `None`. Negation clauses ("No coarctation") leave field blank. |
| `ao_annulus` | `Ao_Annulus` | Numeric / Text | `cm` | 1.2 – 3.5 cm | 0.5 – 5.0 cm | `None` | Missing: `None`. Stored in cm. |
| `ao_root` | `Ao_Root` | Numeric / Text | `cm` | 1.8 – 5.0 cm | 1.0 – 7.0 cm | `None` | Missing: `None`. Stored in cm. |
| `ao_stj` | `Ao_STJ` | Numeric / Text | `cm` | 1.5 – 4.5 cm | 0.8 – 6.5 cm | `None` | Missing: `None`. Stored in cm. |
| `rv_size` | `RV_Size` | Chamber Size | `cm` | 1.0 – 4.5 cm | 0.5 – 6.0 cm | `None` | Missing: `None`. Accepts numeric (cm) or qualitative ("Normal", "Dilated"). |
| `ra_size` | `RA_Size` | Chamber Size | `cm` | 1.5 – 6.0 cm | 0.5 – 8.0 cm | `None` | Missing: `None`. Accepts numeric (cm) or qualitative ("Normal", "Enlarged"). |
| `ivc` | `IVC` | Numeric / Text | `cm` | 0.5 – 3.5 cm | 0.2 – 5.0 cm | `None` | Missing: `None`. Accepts diameter or collapse descriptor. |
| `mv_finding` | `MV` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Stored verbatim (e.g. "Mild MR", "Normal"). |
| `av_finding` | `AV` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Stored verbatim (e.g. "Normal, Trileaflet"). |
| `tv_finding` | `TV` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Stored verbatim (e.g. "Mild TR"). |
| `pv_finding` | `PV` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Stored verbatim (e.g. "Normal"). |
| `e_a_ratio` | `E_A_Ratio` | Numeric / Text | Unitless (`""`) | 0.3 – 3.5 | 0.1 – 6.0 | `None` | Missing: `None`. Dimensionless velocity ratio. |
| `pasp` | `PASP` | Numeric / Text | `mmHg` | 10 – 90 mmHg | 5 – 150 mmHg | `None` | Missing: `None`. Estimated from RVSP or TR gradient. |
| `wall_motion` | `Wall_Motion` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Free text regional/global motion descriptions. |
| `pericardial_effusion` | `Pericardial_Effusion` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Stored verbatim ("None", "Normal", "Mild Effusion"). |
| `clots_thrombus` | `Clots_Thrombus` | Qualitative | N/A | N/A | N/A | `None` | Missing: `None`. Stored verbatim ("Nil", "No Thrombus", "Present"). |
| `relative_wall_thickness` | `RWT` | Numeric / Text | Unitless (`""`) | 0.20 – 0.70 | 0.10 – 1.00 | `None` | Formula: $2 \times \text{PWd} \div \text{LVIDd}$. Computed or extracted. |
| `lv_mass` | `LV_Mass` | Numeric / Text | `g/m²` (LVMI) | 30 – 250 g/m² | 10 – 400 g/m² | `None` | Devereux formula in cm indexed by BSA ($g/m^2$). |
| `lvidd_indexed_value` | `LVIDd_Indexed` | Numeric / Text | `cm/m²` | 1.5 – 4.0 cm/m² | 0.5 – 6.0 cm/m² | `None` | Formula: $\text{LVIDd} \div \text{BSA}$. Flagged if BSA missing. |
| `la_diameter_indexed_value` | `LA_Diameter_Indexed` | Numeric / Text | `cm/m²` | 1.0 – 3.5 cm/m² | 0.5 – 5.5 cm/m² | `None` | Formula: $\text{LA} \div \text{BSA}$. Flagged if BSA missing. |
| `bsa` | `BSA` | Numeric | `m²` | 0.8 – 3.0 m² | 0.4 – 4.0 m² | `None` | From report, DuBois formula ($0.007184 \times \text{Ht}^{0.725} \times \text{Wt}^{0.425}$), or manual entry. |
| `height` | `Height` | Numeric | `cm` | 40 – 250 cm | 20 – 300 cm | `None` | Converted to cm if given in m or inches. |
| `weight` | `Weight` | Numeric | `kg` | 2 – 300 kg | 1 – 500 kg | `None` | Converted to kg if given in lbs. |

---

## 4. Prediction Rule Engine (v4.0 Enhanced)

The clinical prediction engine in [`app/predictor/rules_v4.py`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py) evaluates 18 rule categories and generates evidence-grounded predictions.

### Rule Logic & Threshold Reference

| Category | Predicted Condition | Severity Level | Exact Threshold Logic | Fields Read |
| :--- | :--- | :--- | :--- | :--- |
| **Thrombus** | `Left Ventricular Thrombus` / `Intracardiac Thrombus` | `severe` | `N.contains(p.clots, "thrombus", "clot") and not N.is_absent(p.clots)` or conclusion mentions thrombus. | `clots_thrombus`, `impression_text` |
| **Ischemia / RWMA** | `Regional Wall Motion Abnormality` / `Ischemic Pattern` | `moderate` to `severe` | Segmental RWMA detected (`hypokinesia`, `akinesia`, `dyskinesia`) across Septal, Anterior, Inferior, Lateral, Posterior, or Apical walls, or global `wall_motion` abnormal. | `wall_motion`, `rwma`, segmental wall fields |
| **LV Function** | `Severe LV Systolic Dysfunction` | `severe` | `EF < 35%` | `ef` |
| **LV Function** | `Moderate LV Systolic Dysfunction` | `moderate` | `35% <= EF < 45%` | `ef` |
| **LV Function** | `Mild LV Systolic Dysfunction` | `mild` | `45% <= EF < 55%` | `ef` |
| **LV Function** | `Preserved LVEF` | `normal` | `EF >= 55%` and no RWMA | `ef`, `wall_motion` |
| **Pericardium** | `Pericardial Effusion` | `mild` / `moderate` / `severe` | `not N.is_absent(p.effusion)` and `N.contains(p.effusion, "effusion", "mild", "moderate", "large")`. Graded by severity keyword. | `pericardial_effusion`, `impression_text` |
| **Cardiomyopathy** | `Dilated Cardiomyopathy (DCM)` | `severe` | `EF < 45%` AND `LVIDd > 58 mm` AND `LVIDs > 40 mm`, or stated in conclusion. | `ef`, `lvidd`, `lvids` |
| **Cardiomyopathy** | `Hypertrophic Cardiomyopathy (HCM / HOCM)` | `severe` (stated) / `moderate` (screen) | Stated in conclusion, OR screening: `ivsd_thresh >= 15 mm` (or `13 mm` if age < 40) AND `EF >= 55%`. | `ivsd`, `ivsd_max`, `ef` |
| **Hypertrophy** | `Left Ventricular Hypertrophy (LVH)` | `moderate` | `ivsd_thresh > 11 mm` OR `pwd_thresh > 11 mm` OR `LVMI > 115 g/m²` OR wall thickening. | `ivsd`, `ivsd_max`, `pwd`, `pwd_max`, `lv_mass` |
| **Hypertrophy** | `Severe Left Ventricular Hypertrophy` | `severe` | `ivsd_thresh >= 15 mm` OR `pwd_thresh >= 15 mm` | `ivsd`, `ivsd_max`, `pwd`, `pwd_max` |
| **Hypertrophy** | `Concentric LV Hypertrophy` | `moderate` | `RWT > 0.42` AND (`LVMI > 115 g/m²` OR `ivsd_thresh > 11 mm`) | `relative_wall_thickness`, `lv_mass`, `ivsd` |
| **Hypertrophy** | `Concentric LV Remodeling` | `mild` | `RWT > 0.42` AND normal mass (`LVMI <= 115 g/m²`) AND normal thickness (`IVSd <= 11 mm`) | `relative_wall_thickness`, `lv_mass`, `ivsd` |
| **Hypertrophy** | `Eccentric LV Hypertrophy` | `moderate` | `RWT <= 0.42` AND `LVMI > 115 g/m²` | `relative_wall_thickness`, `lv_mass` |
| **Hypertrophy** | `Asymmetric Septal Hypertrophy (ASH)` | `moderate` | `IVSd / PWd >= 1.3` (or `>= 1.5` in elderly) with septal thickening | `ivsd`, `pwd` |
| **Pulmonary Pressure** | `Pulmonary Hypertension` | `mild` / `moderate` / `severe` | `PASP >= 36 mmHg` (Mild: 36–45, Moderate: 46–60, Severe: >60) OR `TV velocity >= 2.8 m/s`. | `pasp`, `tv_peak_velocity` |
| **Diastolic Function** | `Grade I Diastolic Dysfunction` | `mild` | `E/A ratio < 0.8` (Impaired relaxation) | `e_a_ratio` |
| **Diastolic Function** | `Grade II Diastolic Dysfunction` | `moderate` | `0.8 <= E/A <= 2.0` AND `LA Diameter > 40 mm` (Pseudonormal pattern) | `e_a_ratio`, `la_diameter` |
| **Diastolic Function** | `Grade III Diastolic Dysfunction` | `severe` | `E/A ratio > 2.0` OR restrictive filling pattern in conclusion | `e_a_ratio`, `impression_text` |
| **Valves** | `Aortic / Mitral / Tricuspid / Pulmonary Regurgitation` | `mild` / `moderate` / `severe` | Qualitative finding or abbreviation (`MR`, `AR`, `TR`, `PR`) positive with severity grader. | `mv_finding`, `av_finding`, `tv_finding`, `pv_finding` |
| **Valves** | `Aortic / Mitral / Tricuspid / Pulmonary Stenosis` | `mild` / `moderate` / `severe` | Measured peak gradient/velocity (`AV grad >= 40 mmHg` severe; `MV grad >= 5 mmHg` significant) or text finding. | Valve findings, peak velocity, peak gradient |
| **Valves** | `Mitral Valve Prolapse (MVP)` | `mild` / `moderate` | Finding or conclusion states prolapse / MVP | `mv_finding`, `impression_text` |
| **Valves** | `Multiple Valve Disease` | `moderate` | $\ge 2$ distinct diseased valves identified | All valve fields |
| **Outflow Tract** | `Left Ventricular Outflow Tract Obstruction` | `severe` | `LVOT gradient >= 30 mmHg` OR `LVOT velocity >= 2.7 m/s` OR SAM / LVOTO in conclusion. | `lvot_peak_gradient`, `lvot_peak_velocity`, `impression_text` |
| **Chambers** | `Left Atrial Enlargement (LAE)` | `mild` / `moderate` | `LA Diameter > 40 mm` (or `LA/BSA > 2.2 cm/m²`) OR qualitative enlargement | `la_diameter`, `la_diameter_indexed_value` |
| **Chambers** | `Right Atrial / Ventricular Enlargement` | `moderate` | `RA / RV` size described as enlarged/dilated or numeric threshold exceeded | `ra_size`, `rv_size` |
| **Chambers** | `IVC Plethora / Congestion` | `moderate` | `IVC diameter > 2.1 cm` or reduced respiratory collapse | `ivc` |
| **Aorta** | `Aortic Root Dilation / Ascending Aneurysm` | `moderate` / `severe` | `Ao Root > 38 mm` OR `Ao Diameter > 40 mm` OR `STJ > 36 mm` OR `Annulus > 26 mm` | `ao_root`, `ao_diameter`, `ao_stj`, `ao_annulus` |
| **Combined** | `Hypertensive Heart Disease` | `moderate` | `LVH` or `Concentric Remodeling` present together with `Diastolic Dysfunction` or `LA Enlargement`. | Combined wall & chamber fields |
| **Combined** | `Cor Pulmonale / Right Heart Strain` | `severe` | `Pulmonary Hypertension` present together with `RV Dilation`, `RA Dilation`, `IVC Plethora`, or `Severe TR`. | Combined right heart fields |

---

### Segmental Measurement Routing Confirmation

For reports reporting segmental wall thickness (e.g. `IVS – Basal: 14mm; Mid: 20mm; Apical: 22mm`):
- **Threshold checks (LVH > 11mm, Severe LVH >= 15mm, HCM screening >= 15mm/13mm, Exercise safety contraindications)**: Use the **MAXIMUM** segmental measurement (`ivsd_max = 22 mm`, `pwd_max = 20 mm`).
- **LV Mass and LVMI Calculations**: Strictly use the **BASAL** measurements (`ivsd = 14 mm`, `pwd = 12 mm`) in centimeters ($1.4\text{ cm}, 1.2\text{ cm}, 4.0\text{ cm}$) per ASE standards.

**Exact Code Line in [`app/predictor/rules_v4.py:633-634`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py#L633-L634)**:
```python
ivsd_thresh = max(filter(None, [p.ivsd, getattr(p, "ivsd_max", None)])) if (p.ivsd is not None or getattr(p, "ivsd_max", None) is not None) else None
pwd_thresh = max(filter(None, [p.pwd, getattr(p, "pwd_max", None)])) if (p.pwd is not None or getattr(p, "pwd_max", None) is not None) else None
```

And in [`app/predictor/rules_v4.py:1563`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py#L1563) for Exercise Safety:
```python
ivsd_val = max(filter(None, [p.ivsd, getattr(p, "ivsd_max", None)])) if (p.ivsd is not None or getattr(p, "ivsd_max", None) is not None) else None
```

---

### Confidence, Prediction, and Severity Scoring (§9)

The rule engine calculates **three independent 0–100 scores** for every predicted disease (they do not sum to 100):

1. **Confidence Score (0–100)**: Quantifies evidence provenance and corroboration across independent source classes:
   - `1 source class` (e.g. numeric threshold only) = `48%`
   - `2 source classes` (e.g. numeric threshold + conclusion text) = `68%`
   - `3 source classes` (e.g. numeric + qualitative finding + conclusion) = `84%`
   - `4+ source classes` (multi-parameter corroboration) = `95%`
   - *Caps*: Capped at `35%` if derived from manual doctor dropdown; capped at `45%` if derived from an estimated fallback chain.
2. **Prediction Score (0–100)**: Quantifies how far past the age-resolved upper normal limit a parameter sits ($0\% = 45$, saturation at $+50\%$ excess $= 100$).
3. **Severity Score (0–100)**: Direct translation of clinical disease grade (`Normal` = 0, `Mild` = 25, `Moderate` = 55, `Severe` = 90, `Critical/Life-Threatening` = 100).

---

## 5. Known Gaps & Actionable TODOs

The following gaps and opportunities have been audited against the codebase with exact file and line references:

### 1. Mitral Valve Area (MVA) Stenosis Grading Rule
- **Issue**: `mv_area` is extracted and stored in `CardiacReport` ([`app/models.py:123`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/models.py#L123)) and mapped in [`rules_v4.py:58`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py#L58), but [`_rule_valves()`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py#L871-L915) currently grades Mitral Stenosis using peak gradient (`mv_grad`) and peak velocity (`mv_vel`) without evaluating the anatomical MVA cut-offs.
- **Actionable TODO**: In `app/predictor/rules_v4.py`, update `_rule_valves()` to add MVA thresholds ($>1.5\text{ cm}^2$ = Mild, $1.0 - 1.5\text{ cm}^2$ = Moderate, $<1.0\text{ cm}^2$ = Severe Mitral Stenosis).

### 2. Stroke Volume and Cardiac Output from LVOT VTI
- **Issue**: `lvot_vti` is extracted ([`app/models.py:122`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/models.py#L122)) and stored, but no downstream rule calculates Stroke Volume ($\text{SV} = \pi \times (\text{LVOT Diameter} / 2)^2 \times \text{LVOT VTI}$) or Cardiac Index.
- **Actionable TODO**: In `app/ocr/bsa.py` and `app/predictor/rules_v4.py`, add hemodynamic calculation for SV and Cardiac Index when `ao_annulus` (or LVOT diameter) and `lvot_vti` are both present.

### 3. Wall Thickening Fraction (Systolic Function Auxiliary)
- **Issue**: `ivss` and `pws` are captured and stored ([`app/models.py:88-89`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/models.py#L88-L89)), but systolic thickening percentage ($\Delta \text{IVS} = (\text{IVSs} - \text{IVSd}) / \text{IVSd} \times 100$) is not mapped to an active ischemic/viability rule in `_rule_ischemia()`.
- **Actionable TODO**: In `app/predictor/rules_v4.py`, add septal/posterior wall systolic thickening index ($<30\%$ indicates impaired regional contractility).

### 4. Advanced Diastolic Grading (Tissue Doppler e' and E/e' Ratio)
- **Issue**: `_rule_diastolic()` ([`app/predictor/rules_v4.py:735-763`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/predictor/rules_v4.py#L735-L763)) grades diastolic dysfunction using mitral E/A ratio and LA diameter. Modern ASE/EACVI guidelines also utilize septal/lateral mitral annular tissue Doppler ($e'$) and $E/e'$ ratio.
- **Actionable TODO**: Add `e_prime_septal`, `e_prime_lateral`, and `e_e_prime_ratio` to `app/models.py` and `app/ocr/parameter_dict.py`, and expand `_rule_diastolic()` to evaluate $E/e' > 14$ as positive evidence of elevated LV filling pressures.

### 5. Annulus Dilation Rules
- **Issue**: Annulus diameters (`mv_annulus`, `tv_annulus`, `pv_annulus`, `ao_annulus`) are in the schema ([`app/models.py:96,124-126`](file:///c:/Users/ASUS/Downloads/OCRRR/Sushrut%20-%20MVP/cardiac_ocr_system/backend/app/models.py#L96)), but functional annular dilation (e.g. Tricuspid Annulus $>40\text{ mm}$ or $>21\text{ mm/m}^2$) does not trigger a dedicated remodeling finding.
- **Actionable TODO**: In `app/predictor/rules_v4.py: _rule_valves()`, add checks for tricuspid/mitral annular dilation supporting functional regurgitation.
