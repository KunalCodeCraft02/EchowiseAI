# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sushrut MVP: a FastAPI backend that turns uploaded 2D Echocardiogram reports (PDF / scanned image /
phone photo) into structured clinical parameters, runs a deterministic disease-prediction rule
engine over them, and generates AI exercise/rehab plans. There is no separate frontend build —
`backend/app/static/` (plain HTML/CSS/JS) is served directly by FastAPI via `StaticFiles` and a few
`FileResponse` routes in `app/main.py`.

A full architecture write-up with exact function/line references already exists at
`../overview.md` (one level above this directory) — read it before making non-trivial changes to
the OCR/extraction/rules pipeline. `backend/current_logic_review.md` is a supplementary,
point-in-time code walkthrough of the same pipeline.

## Commands

All commands run from `backend/`, using the checked-in venv (Windows paths):

```
./venv/Scripts/python.exe run.py                     # start dev server (uvicorn, reload) on http://127.0.0.1:8000
./venv/Scripts/python.exe -m pytest                   # run the full test suite
./venv/Scripts/python.exe -m pytest tests/test_rules_v4.py            # single test file
./venv/Scripts/python.exe -m pytest tests/test_rules_v4.py::test_name # single test
./venv/Scripts/python.exe -m pip install -r requirements.txt          # install deps
```

Extraction accuracy harness (scores the real pipeline against
`tests/fixtures/ground_truth.json`, separate from pytest):

```
./venv/Scripts/python.exe tools/accuracy_report.py                # full run (OCR + Groq)
./venv/Scripts/python.exe tools/accuracy_report.py --no-llm        # deterministic/offline only
./venv/Scripts/python.exe tools/accuracy_report.py --no-cache      # force re-OCR, ignore .ocr_cache
./venv/Scripts/python.exe tools/accuracy_report.py --save baseline.json
./venv/Scripts/python.exe tools/accuracy_report.py --diff baseline.json
```

It reports `correct` / `WRONG` / `FALSE POSITIVE` / `missed` per parameter. WRONG and FALSE
POSITIVE are the dangerous categories (a wrong value silently feeds the rule engine); trading a
`missed` for a `WRONG` is a regression even if the headline percentage improves.

Secrets (`GROQ_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, etc.) live in `backend/.env`
(gitignored) and are loaded by `app/config.py` via `python-dotenv`.

## Architecture

### Request flow

`app/main.py` wires two routers: `auth_router` (`/api/auth/...`) and `reports`
(`/api/reports/...`), plus page routes that just return static HTML
(`app/static/pages/*.html`). Auth is JWT bearer tokens (`app/auth.py`, PBKDF2-HMAC-SHA256
password hashing via stdlib `hashlib`, no bcrypt/passlib). `app/database.py` is SQLite via
SQLAlchemy, with a dependency-free "migration" (`_migrate_missing_columns`) that ALTERs in any
model column missing from an existing `cardiac.db` on startup — there is no Alembic.

### The OCR → prediction pipeline (`app/ocr/pipeline.py:process_report`)

Upload triggers a background task that walks these stages, writing `progress` /
`progress_stage` onto the `CardiacReport` row after each one so the frontend can poll
`GET /api/reports/{uid}/progress`:

1. **Ingestion & document check** (`app/routers/reports.py`, `app/ocr/document_check.py`) —
   rejects non-PDF/image uploads and flags non-echo documents as `failure_reason="wrong_file"`
   before any OCR is attempted.
2. **Preprocessing** (`app/ocr/preprocessing.py`) — digital PDFs are detected and read directly
   via `pdfplumber` (no OCR, no API call); everything else is rasterized/loaded as images.
3. **Geometric OCR** (`app/ocr/text_extraction.py`) — PaddleOCR is the primary reader; it
   returns bounding-box geometry that is clustered locally into rows/columns to reconstruct
   table structure (`OCR_ROW_OVERLAP_RATIO`, `OCR_COLUMN_GAP_RATIO` in `config.py`). Falls back
   to Tesseract, then flat-line OCR, if PaddleOCR is unavailable. No cloud Vision/Document AI is
   used in the live path (`document_ai_extractor.py` exists but is dormant — see below).
4. **Semantic field parsing** (`app/ocr/extractor.py`, `parameter_dict.py`) — a deterministic
   dictionary/regex matcher runs first (free, offline, reproducible); `app/ocr/groq_extractor.py`
   is called **text-only** (never given the image) to resolve unmapped label variants and read
   narrative findings out of prose sections (Conclusion/Impression/Findings/Valves/Summary).
5. **Indexed hemodynamics** (`app/ocr/bsa.py`) — BSA (from report, DuBois formula, or manual
   entry) drives Devereux LV Mass, LVMI, and indexed chamber diameters; these are cleared and
   flagged for manual entry when BSA is missing.
6. **Cross-validation** (`app/ocr/cross_validation.py`) — corroborates Groq-sourced values
   against the independent PaddleOCR read; this is what real confidence scores are derived from
   (Groq itself returns none — see `GROQ_CORROBORATED_CONFIDENCE` / `GROQ_UNCORROBORATED_CONFIDENCE`).
7. **Persistence** — the full result is stored on `CardiacReport` (`app/models.py`, 45+
   parameter columns) and mirrored as an audit JSON in `backend/extractions/<report_uid>.json`
   via `app/ocr/export.py` (`EXTRACTION_JSON_DIR`). Extraction JSON and `app/uploads/` contain
   real patient data — never commit them (already gitignored).

Every stage is designed to **degrade, not crash or invent data**: a field that can't be read
confidently is stored as `None`/empty and added to `flagged_params` with a `flag_reason`; it is
never guessed at.

### Prediction rule engine (`app/predictor/rules_v4.py`, `normalize_v4.py`)

`evaluate_v4()` is a deterministic clinical decision-support engine (18 rule categories,
documented with exact thresholds in `../overview.md` §4) triggered via
`POST /api/reports/{uid}/predict`. The one invariant that matters most across the whole engine
(see the docstring in `tests/test_rules_v4.py`): **missing data is never imputed as normal**
("G1" — Missing Never Becomes Normal). An empty/partial report must never produce a "Normal
Heart" or "Exercise Cleared" verdict; unfilled parameters are unassessed, not passing.

For segmental wall-thickness reporting (IVS/PW given as Basal/Mid/Apical), threshold checks
(LVH, HCM screening, exercise-safety contraindications) use the **maximum** segmental value,
while LV Mass/LVMI calculations use strictly the **basal** value in cm — these must not be
conflated (`rules_v4.py` around line 633 and the exercise-safety block around line 1563).

Each predicted disease carries three independent 0–100 scores that do **not** sum to
100 — Confidence (evidence provenance/corroboration), Prediction (how far past the age-resolved
normal limit), Severity (clinical grade) — computed per `../overview.md` §"Confidence,
Prediction, and Severity Scoring".

`app/predictor/value_table.py` supplies age/sex-resolved normal defaults that the **review UI**
(not the extractor, not the rule engine) uses to pre-populate empty cards
(`initAutoNormalDefaults()` in `report.html`) — this is a UI-only convenience layer, distinct
from and never fed back into extraction or prediction.

### Care plan generation (`app/predictor/groq_client.py`, `rehab_generator.py`)

`POST /api/reports/{uid}/care-plans` generates disease-specific exercise/rehab plans via Groq,
gated strictly by the rule engine's `exercise_safety` verdict:
- `Exercise Contraindicated` → no LLM call at all; a fixed message is returned instead of a plan
  (mixing "plan" and "do not exercise" in one response is treated as a display-worthy failure
  mode, not just a wording choice).
- `Indeterminate` (a compulsory parameter group is missing) → also blocked, no LLM call.
- No disease detected → general guidance instead of a disease-specific plan.
See the message constants and `_plan_error_message()` at the top of `app/routers/reports.py` —
raw provider error bodies are logged server-side keyed by `report_uid` and never shown to the
clinician.

### Config (`app/config.py`)

Single source of truth for all tunables, read from env vars with local defaults, loaded from
`backend/.env`. Notable non-obvious bits documented inline there:
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is forced at import time (before anything
  imports `google.protobuf`) because PaddleOCR's bundled protobuf files break under the C++/upb
  descriptor implementation — this must stay the very first thing `config.py` does, since
  `config` is the first module every entry point imports.
- `protobuf==3.20.2` in `requirements.txt` is pinned hard — paddlepaddle 2.6.2 requires
  `protobuf<=3.20.2` on Windows; do not unpin it.
- `google-cloud-documentai` is intentionally **not** installed — it requires `protobuf>=4.25`
  and cannot coexist with paddlepaddle. `app/ocr/document_ai_extractor.py` exists but is dormant
  (lazy imports only); nothing in the live pipeline calls it.
- Groq is used two ways that must not be confused: `GROQ_VISION_MODEL` (dormant, unused by the
  live pipeline) vs. `GROQ_SEMANTIC_MODEL` (the live, text-only label/narrative resolver). Set
  `GROQ_SEMANTIC_ENABLED=0` to run extraction fully offline/deterministic.
- OCR confidence/thresholds are all on a 0–100 scale throughout the codebase (PaddleOCR's native
  0–1 is multiplied by 100 at the single normalization point in `text_extraction.py`) — never mix
  scales when touching this code.
- Several OCR-tuning constants (`OCR_MIN_LONG_SIDE`, `OCR_DET_UNCLIP_RATIO`,
  `OCR_COLUMN_GAP_RATIO`, `OCR_DROP_SCORE`, etc.) record measured accuracy tradeoffs in comments
  from sweeps against `tests/fixtures/ground_truth.json` — read the comment before changing a
  default, the "obvious" improvement has usually already been tried and measured worse.

### Known gaps

`../overview.md` §5 lists specific unimplemented rules with exact TODOs (MVA stenosis grading,
stroke volume/cardiac index from LVOT VTI, wall-thickening fraction, E/e' diastolic grading,
annulus dilation rules) — check there before assuming a clinical parameter that's extracted and
stored is also acted on by the rule engine; several are captured but not yet wired into
`rules_v4.py`.
