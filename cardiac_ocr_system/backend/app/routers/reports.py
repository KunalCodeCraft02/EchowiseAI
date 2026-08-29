import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.auth import get_current_doctor
from app.config import UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_MB
from app.ocr.parameter_dict import FIELD_LABELS, REPORT_SECTIONS
from app.ocr.pipeline import process_report
from app.predictor.rules_v4 import evaluate_v4, COMPULSORY_GROUPS, combined_guidance_summary
from app.predictor.value_table import dropdown_config
from app.predictor.groq_client import (GroqError, generate_care_plan,
                                       generate_disease_care_plans,
                                       general_population_guidance)
from app.predictor.rehab_generator import generate_unified_rehab_plan

router = APIRouter(prefix="/api/reports", tags=["reports"])

logger = logging.getLogger(__name__)

# Shown INSTEAD of any exercise plan when the rule engine says exercise is contraindicated. The
# LLM is not called at all in that case: a plan that exists can be read, skimmed or forwarded,
# and "here is your exercise plan, but do not exercise" is exactly the kind of mixed message a
# patient acts on the wrong half of.
EXERCISE_BLOCKED_MESSAGE = (
    "Exercise is not recommended for this patient based on the findings above. "
    "Cardiology clearance required before any activity plan can be generated."
)

# Shown when the rule engine matched no abnormal finding. Worded as the result it is, not as a
# failure: there is no disease to write a plan for, which is the best possible answer.
NO_DISEASE_MESSAGE = (
    "No cardiac disease was detected on this report, so there is no condition-specific exercise "
    "plan to generate. General activity advice remains at the treating physician's discretion."
)

# Exercise Safety = "Indeterminate" is not a green light: the hard-stop screen could not be
# completed at all, so no plan -- disease-specific OR general -- is generated. The LLM is never
# called in this case either, matching the Contraindicated branch.
EXERCISE_INDETERMINATE_MESSAGE = (
    "Exercise recommendations cannot be finalized: the Exercise Safety screen could not be "
    "completed because one or more of the nine compulsory parameter groups is missing. Complete "
    "the missing parameters on the review page and re-run the prediction before generating an "
    "exercise plan."
)

# No cardiac disease was predicted, but at least one filled parameter sits mildly outside the
# ideal range (§5) -- a third output state, distinct from both the default general guidance and a
# disease-specific plan.
BORDERLINE_NOTICE = (
    "No cardiac disease was found on this echocardiogram. However, one or more parameters are "
    "mildly outside the ideal range for this patient's age group -- see the parameter "
    "improvement guidance below."
)


def _plan_error_message(exc: GroqError, report_uid: str) -> str:
    """Log the technical failure; return the sentence the clinician is shown.

    THE POINT: nothing derived from the upstream response ever reaches the page. A raw provider
    error body is JSON, and JSON on a clinical screen reads as a crash -- it tells a doctor
    nothing they can act on while implying the report itself is broken. The status code, body and
    traceback go to the server log, keyed by report_uid so a support request can still be traced.
    """
    logger.error("Care plan generation failed for report %s: %s", report_uid, exc, exc_info=exc)
    return exc.user_message


def _exercise_verdict_name(report: models.CardiacReport):
    verdict = (report.predicted_diseases or {}).get("exercise_safety") or {}
    return verdict.get("cardiac_disease_name")


def _exercise_contraindicated(report: models.CardiacReport) -> bool:
    return _exercise_verdict_name(report) == "Exercise Contraindicated"


def _compulsory_findings(report: models.CardiacReport) -> dict:
    """Every compulsory-group field that is actually filled, labelled by group -- passed to the
    exercise-plan generator so it can reference EF/PASP/etc. even when they did not themselves
    trigger a disease (§4)."""
    out = {}
    for group_name, fields in COMPULSORY_GROUPS:
        for f in fields:
            value = getattr(report, f, None)
            if value:
                out.setdefault(group_name, {})[f] = value
    return out


def _set_care_plans_run(db: Session, report: models.CardiacReport, status: str,
                        done: int = 0, total: int = 0, current: str = "",
                        error: str = None) -> None:
    """Publish the care-plan run's measured state so the report page can poll it.

    Committed on every call: the polling request reads through its OWN session, so an uncommitted
    update is a percentage that never moves. These writes are small and happen once per disease,
    not per tick.

    `progress` is computed from done/total and NOTHING else -- never from elapsed time. It also
    never reaches 100 until the run is actually finished, so the bar cannot claim completion while
    the last Groq call is still outstanding.
    """
    report.care_plans_run = {
        "status": status,
        "done": done,
        "total": total,
        "progress": int(round(100 * done / total)) if total else (100 if status == "done" else 0),
        "current": current or None,
        "error": error,
    }
    db.add(report)
    db.commit()


@router.post("/upload", response_model=schemas.ReportCreateResponse, status_code=status.HTTP_201_CREATED)
def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_name: str = Form(...),
    patient_age: str = Form(None),
    patient_gender: str = Form(None),
    is_athlete: bool = Form(False),
    db: Session = Depends(get_db),
    current_doctor: models.Doctor = Depends(get_current_doctor),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    saved_name = f"{uuid.uuid4().hex}{suffix}"
    dest_path = UPLOAD_DIR / saved_name

    with dest_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f} MB). Max {MAX_UPLOAD_SIZE_MB} MB.")

    report = models.CardiacReport(
        doctor_id=current_doctor.id,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        is_athlete=bool(is_athlete),
        original_filename=file.filename,
        status="uploaded",
        progress=0,
        progress_stage="Queued",
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    background_tasks.add_task(_run_pipeline_task, report.id, str(dest_path))

    return schemas.ReportCreateResponse(
        report_uid=report.report_uid,
        id=report.id,
        status=report.status,
        message="Upload received. Processing started.",
    )


def _run_pipeline_task(report_id: int, file_path: str):
    """Runs in a background task with its own DB session (request session is closed by then)."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        report = db.query(models.CardiacReport).filter(models.CardiacReport.id == report_id).first()
        if report:
            process_report(db, report, file_path)
    finally:
        db.close()


def _field_sources(report: models.CardiacReport) -> dict:
    """{db_field: "doctor_dropdown" | "doctor_custom" | "ocr"} from the stored audit trail.

    Fields the pipeline extracted have no doctor-entered source and are reported as "ocr", which
    is what the scorer treats as a real measurement.
    """
    sources = {}
    for field, entry in (report.extraction_meta or {}).items():
        if isinstance(entry, dict) and entry.get("source") in ("doctor_dropdown", "doctor_custom"):
            sources[field] = entry["source"]
    return sources


def _build_params_for_evaluation(report: models.CardiacReport) -> dict:
    """Build the parameters dict for evaluate_v4, unpacking segmental wall thickness and metadata."""
    params = {f: getattr(report, f) for f in models.CardiacReport.PARAM_FIELDS}
    params["impression_text"] = report.impression_text

    meta = report.extraction_meta or {}
    for f_key in ("ivsd", "pwd"):
        f_meta = meta.get(f_key) or {}
        if isinstance(f_meta, dict):
            for k in (f"{f_key}_basal", f"{f_key}_mid", f"{f_key}_apical", f"{f_key}_max"):
                if k in f_meta:
                    params[k] = f_meta[k]
            if f_meta.get("segment_values_mm"):
                seg_vals = f_meta["segment_values_mm"]
                if "max" in seg_vals and f"{f_key}_max" not in params:
                    params[f"{f_key}_max"] = seg_vals["max"]
                for seg_name in ("basal", "mid", "apical"):
                    if seg_name in seg_vals and f"{f_key}_{seg_name}" not in params:
                        params[f"{f_key}_{seg_name}"] = seg_vals[seg_name]
    return params


def _get_owned_report(db: Session, report_uid: str, doctor: models.Doctor) -> models.CardiacReport:
    report = db.query(models.CardiacReport).filter(models.CardiacReport.report_uid == report_uid).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    if report.doctor_id != doctor.id:
        raise HTTPException(status_code=403, detail="You do not have access to this report.")
    return report


@router.get("/meta/sections")
def get_sections():
    """The review page's section layout, served from parameter_dict.py.

    The frontend used to hardcode its own field list, which is how a parameter could be
    extracted correctly and still never appear on screen: "Aortic Root" was stored in ao_root
    while the page only rendered ao_diameter (labelled, confusingly, "Aortic Root Diameter").
    Serving the layout from the dictionary means a parameter added there shows up automatically
    and the two cannot disagree.

    `dropdowns` carries the same guarantee for the blank-field dropdowns: the option list, the
    unit and the already-resolved value for every age group come from value_table.py, so the page
    can never offer an option the engine cannot resolve or show a band the tables have moved on
    from. It is static per deployment and therefore fetched once, with the sections.
    """
    return {"sections": REPORT_SECTIONS, "labels": FIELD_LABELS, "dropdowns": dropdown_config()}


@router.get("/{report_uid}/progress", response_model=schemas.ProgressOut)
def get_progress(report_uid: str, db: Session = Depends(get_db),
                  current_doctor: models.Doctor = Depends(get_current_doctor)):
    report = _get_owned_report(db, report_uid, current_doctor)
    return schemas.ProgressOut(
        report_uid=report.report_uid,
        status=report.status,
        progress=report.progress,
        progress_stage=report.progress_stage,
        failure_reason=report.failure_reason,
    )


@router.get("/{report_uid}", response_model=schemas.ReportOut)
def get_report(report_uid: str, db: Session = Depends(get_db),
                current_doctor: models.Doctor = Depends(get_current_doctor)):
    report = _get_owned_report(db, report_uid, current_doctor)
    return schemas.ReportOut(**report.to_dict())


@router.put("/{report_uid}", response_model=schemas.ReportOut)
def update_report(report_uid: str, payload: schemas.ReportUpdate, db: Session = Depends(get_db),
                   current_doctor: models.Doctor = Depends(get_current_doctor)):
    report = _get_owned_report(db, report_uid, current_doctor)

    valid_fields = set(models.CardiacReport.PARAM_FIELDS)
    for field, value in payload.parameters.items():
        if field not in valid_fields:
            raise HTTPException(status_code=400, detail=f"Unknown parameter field '{field}'.")
        setattr(report, field, value)

    # Provenance for doctor-filled fields, merged into the per-field audit trail the extraction
    # pipeline already writes. Recorded rather than inferred, because the prediction engine caps
    # the confidence of a finding that rests on a chosen band instead of a measured value, and it
    # cannot tell the two apart from the stored string alone.
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

    # BSA resolution and live re-calculation for indexed fields
    from app.ocr.bsa import resolve_bsa, compute_indexed_fields, calculate_dubois_bsa

    if "bsa" in payload.parameters:
        new_bsa = (payload.parameters["bsa"] or "").strip()
        if new_bsa:
            report.bsa_source = "manual entry"
        else:
            report.bsa_source = "missing"

    if ("height" in payload.parameters or "weight" in payload.parameters) and report.bsa_source != "manual entry":
        h_str = getattr(report, "height", "") or ""
        w_str = getattr(report, "weight", "") or ""
        if h_str and w_str:
            try:
                calc = calculate_dubois_bsa(float(h_str), float(w_str))
                if calc is not None:
                    report.bsa = str(calc)
                    report.bsa_source = "calculated"
            except (ValueError, TypeError):
                pass

    current_params = {f: getattr(report, f) for f in models.CardiacReport.PARAM_FIELDS}
    bsa_val, bsa_source_key, bsa_tag = resolve_bsa(current_params, bsa_source_override=report.bsa_source)
    if bsa_val is not None:
        report.bsa = str(bsa_val)
    report.bsa_source = bsa_source_key

    indexed = compute_indexed_fields(current_params, bsa_val)
    for idx_field in ("lvidd_indexed_value", "la_diameter_indexed_value", "lv_mass"):
        if bsa_val is not None:
            if indexed.get(idx_field) and idx_field not in payload.parameters:
                setattr(report, idx_field, indexed[idx_field])
        else:
            if idx_field not in payload.parameters:
                setattr(report, idx_field, "")

    if payload.confirmed_fields:
        for idx_field in ("lvidd_indexed_value", "la_diameter_indexed_value", "lv_mass"):
            if idx_field in payload.confirmed_fields:
                cur_bsa = getattr(report, "bsa", None)
                if not cur_bsa or not str(cur_bsa).strip():
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot confirm field '{idx_field}': Requires manual BSA entry."
                    )
        confirmed = set(report.doctor_confirmed or [])
        confirmed.update(payload.confirmed_fields)
        report.doctor_confirmed = list(confirmed)
        flagged = set(report.flagged_params or [])
        flagged -= confirmed
        report.flagged_params = list(flagged)

    if payload.extra_findings is not None:
        report.extra_findings = payload.extra_findings

    if payload.is_athlete is not None:
        report.is_athlete = payload.is_athlete

    report.status = "reviewed" if not (report.flagged_params or []) else report.status
    db.add(report)
    db.commit()
    db.refresh(report)
    return schemas.ReportOut(**report.to_dict())


@router.get("", response_model=list[schemas.ReportOut])
def list_reports(db: Session = Depends(get_db), current_doctor: models.Doctor = Depends(get_current_doctor)):
    reports = (
        db.query(models.CardiacReport)
        .filter(models.CardiacReport.doctor_id == current_doctor.id)
        .order_by(models.CardiacReport.created_at.desc())
        .all()
    )
    return [schemas.ReportOut(**r.to_dict()) for r in reports]


@router.post("/{report_uid}/predict", response_model=schemas.PredictionOut)
def predict_disease(report_uid: str, db: Session = Depends(get_db),
                     current_doctor: models.Doctor = Depends(get_current_doctor)):
    """Runs the 36-rule Cardiac Disease Prediction Engine against this report's
    extracted parameters and persists the result."""
    report = _get_owned_report(db, report_uid, current_doctor)

    # v4.0 engine. impression_text is passed as conclusion_text -- the v4.0 spec treats the
    # cardiologist's own conclusion as a first-class trigger (G3), not just as prose.
    params = _build_params_for_evaluation(report)
    # patient_age selects the age band whose thresholds the compulsory-group rules apply. It is
    # the report's own free-text field; an unusable value simply leaves the standard numbers.
    # patient_gender picks the sex-specific limb of a threshold where one is published, and
    # field_sources tells the scorer which values the doctor chose from a dropdown rather than
    # the report having stated them.
    result = evaluate_v4(params, patient_age=report.patient_age,
                         patient_sex=report.patient_gender,
                         field_sources=_field_sources(report),
                         is_athlete=bool(report.is_athlete))

    report.predicted_diseases = result
    report.risk_level = result["risk_level"]
    report.rules_evaluated = result["parameters_available"]
    report.prediction_generated_at = datetime.utcnow()
    # A fresh prediction invalidates any previously generated care plan.
    report.rehab_plan = None
    report.diet_plan = None
    report.recommendation_error = None
    report.recommendation_generated_at = None
    report.care_plans = None
    report.care_plans_generated_at = None
    # The old run's progress goes with its plans. Left behind, a completed "7 of 7" would sit
    # against a prediction whose plans no longer exist.
    report.care_plans_run = None
    # Saved physician feedback survives a re-run; anything the new result no longer predicts is
    # flagged stale rather than dropped.
    _sync_confirmation_staleness(db, report,
                                 [d["cardiac_disease_name"] for d in result["diseases"]])
    db.add(report)
    db.commit()

    return schemas.PredictionOut(
        report_uid=report.report_uid,
        risk_level=result["risk_level"],
        rules_total=result["parameters_total"],
        rules_evaluated=result["parameters_available"],
        normal_heart=result["normal_heart"],
        diseases=result["diseases"],
        athlete_screening=result["athlete_screening"],
        exercise_safety=result["exercise_safety"],
        athlete_gray_zone=result["athlete_gray_zone"],
        borderline_parameters=result["borderline_parameters"],
        age_band=result["age_band"],
        age_band_label=result["age_band_label"],
        age_group=result["age_group"],
        age_group_label=result["age_group_label"],
        age_unknown=result["age_unknown"],
        pending_pediatric_review=result["pending_pediatric_review"],
        pediatric_notice=result["pediatric_notice"],
        compulsory_coverage=result["compulsory_coverage"],
        risk_summary=result["risk"],
        disclaimer=result["disclaimer"],
        generated_at=report.prediction_generated_at.isoformat(),
    )


@router.post("/{report_uid}/recommend", response_model=schemas.RecommendationOut)
def recommend_care_plan(report_uid: str, db: Session = Depends(get_db),
                         current_doctor: models.Doctor = Depends(get_current_doctor),
                         mode: Optional[str] = Query(
                             "general",
                             description="Rehab plan tone: 'general' (short, non-disease-specific "
                                         "conversational summary -- the default for this legacy "
                                         "endpoint), 'detailed' (patient-facing daily guide), or "
                                         "any other value for the original clinical F.I.T.T. plan."
                         )):
    """Passes the extracted parameters + predicted disease(s) to Groq to generate one combined,
    GENERAL rehabilitation exercise plan (legacy). Deliberately brief and non-disease-specific --
    see /care-plans for the detailed, disease-specific patient guide. Exercise only -- no dietary
    guidance is requested or returned."""
    report = _get_owned_report(db, report_uid, current_doctor)

    if not report.predicted_diseases:
        raise HTTPException(status_code=400, detail="Run the prediction engine before generating a care plan.")

    # Hard stop before the LLM is reached, not a filter on what it returned.
    if _exercise_contraindicated(report):
        report.rehab_plan = None
        report.diet_plan = None
        report.recommendation_error = None
        report.recommendation_generated_at = datetime.utcnow()
        db.add(report)
        db.commit()
        return schemas.RecommendationOut(
            report_uid=report.report_uid,
            exercise_blocked=True,
            blocked_reason=EXERCISE_BLOCKED_MESSAGE,
            generated_at=report.recommendation_generated_at.isoformat(),
        )

    patient_context = {
        "patient_age": report.patient_age,
        "patient_gender": report.patient_gender,
        "age_band": report.predicted_diseases.get("age_band_label"),
        "extracted_parameters": {f: getattr(report, f) for f in models.CardiacReport.PARAM_FIELDS},
        "risk_level": report.risk_level,
        "normal_heart": report.predicted_diseases.get("normal_heart"),
        "predicted_diseases": report.predicted_diseases.get("diseases", []),
        "athlete_screening": report.predicted_diseases.get("athlete_screening"),
        "exercise_safety": report.predicted_diseases.get("exercise_safety"),
    }

    try:
        plan_markdown = generate_unified_rehab_plan(
            patient_context=patient_context,
            predicted_conditions=report.predicted_diseases.get("diseases", []),
            safety_tier=report.predicted_diseases.get("exercise_safety"),
            mode=mode,
        )
    except Exception as exc:
        logger.error(f"Error in recommend_care_plan: {exc}")
        from app.predictor.rehab_generator import _build_deterministic_fallback, route_conditions, _resolve_safety_tier
        primary_c, secondary_c = route_conditions(report.predicted_diseases.get("diseases", []))
        tier_obj = _resolve_safety_tier(report.predicted_diseases.get("exercise_safety"), patient_context, primary_c)
        plan_markdown = _build_deterministic_fallback(patient_context, primary_c, secondary_c, tier_obj)

    report.rehab_plan = plan_markdown
    report.diet_plan = None
    report.recommendation_error = None
    report.recommendation_generated_at = datetime.utcnow()
    db.add(report)
    db.commit()

    return schemas.RecommendationOut(
        report_uid=report.report_uid,
        rehab_plan=report.rehab_plan,
        key_precautions=None,
        generated_at=report.recommendation_generated_at.isoformat(),
    )


# ===========================================================================================
# Clinician feedback loop (v4.0)
# ===========================================================================================
def _sync_confirmation_staleness(db: Session, report: models.CardiacReport,
                                 current_names: List[str]) -> None:
    """Reconcile saved feedback with a fresh prediction run.

    Disease names encode severity, so correcting a parameter and re-predicting can rename a
    disease ("Moderate Mitral Regurgitation" -> "Mild ..."). A confirmation for a name that is
    no longer produced is MARKED STALE, never deleted: the physician's note is the most valuable
    data in this system, and silently discarding it because a number moved would be a real loss.
    A disease that reappears has its stale flag cleared.
    """
    live = {n.strip().lower() for n in current_names}
    for row in db.query(models.PredictionConfirmation).filter_by(report_id=report.id).all():
        row.is_stale = row.disease_name.strip().lower() not in live
        db.add(row)


@router.get("/{report_uid}/confirmations", response_model=List[schemas.ConfirmationOut])
def list_confirmations(report_uid: str, db: Session = Depends(get_db),
                       current_doctor: models.Doctor = Depends(get_current_doctor)):
    report = _get_owned_report(db, report_uid, current_doctor)
    rows = db.query(models.PredictionConfirmation).filter_by(report_id=report.id).all()
    return [schemas.ConfirmationOut(**r.to_dict()) for r in rows]


@router.post("/{report_uid}/confirmations", response_model=schemas.ConfirmationOut)
def save_confirmation(report_uid: str, payload: schemas.ConfirmationIn,
                      db: Session = Depends(get_db),
                      current_doctor: models.Doctor = Depends(get_current_doctor)):
    """Upsert the physician's verdict and notes for one predicted disease.

    Partial by design: sending only `confirmed` leaves existing notes untouched and vice versa,
    so the toggle and the textarea can save independently without overwriting each other.
    """
    report = _get_owned_report(db, report_uid, current_doctor)
    name = payload.disease_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="disease_name is required.")

    row = (db.query(models.PredictionConfirmation)
             .filter_by(report_id=report.id, disease_name=name).one_or_none())
    if row is None:
        row = models.PredictionConfirmation(report_id=report.id, doctor_id=current_doctor.id,
                                            disease_name=name)
    if payload.confirmed is not None:
        row.confirmed = payload.confirmed
    if payload.clinician_notes is not None:
        row.clinician_notes = payload.clinician_notes
    row.doctor_id = current_doctor.id
    db.add(row)
    db.commit()
    db.refresh(row)
    return schemas.ConfirmationOut(**row.to_dict())


@router.get("/{report_uid}/care-plans/progress", response_model=schemas.CarePlanProgressOut)
def get_care_plans_progress(report_uid: str, db: Session = Depends(get_db),
                            current_doctor: models.Doctor = Depends(get_current_doctor)):
    """Measured progress of the care-plan run, polled WHILE the POST above is still running.

    The two requests are concurrent by design: the POST blocks for the whole generation, and this
    reads the state it commits after each disease, through its own session. `done`/`total` are
    counts of finished Groq calls, so "3 of 7" means three plans genuinely exist.
    """
    report = _get_owned_report(db, report_uid, current_doctor)
    run = report.care_plans_run or {}
    return schemas.CarePlanProgressOut(
        report_uid=report.report_uid,
        status=run.get("status") or "idle",
        progress=int(run.get("progress") or 0),
        done=int(run.get("done") or 0),
        total=int(run.get("total") or 0),
        current=run.get("current"),
        error=run.get("error"),
    )


@router.post("/{report_uid}/care-plans", response_model=schemas.CarePlanOut)
def generate_care_plans(report_uid: str, db: Session = Depends(get_db),
                        current_doctor: models.Doctor = Depends(get_current_doctor),
                        mode: Optional[str] = Query(
                            "detailed",
                            description="Rehab plan tone: 'detailed' (patient-facing, disease-"
                                        "specific rehabilitation guide -- the default for this "
                                        "'Predict Exercise based on Disease' endpoint), 'general' "
                                        "(short conversational summary), or any other value for "
                                        "the original clinical F.I.T.T. plan."
                        )):
    """One Groq-generated, DETAILED and disease-specific rehabilitation exercise plan, as
    patient-friendly plain text ("Predict Exercise based on Disease"). See /recommend for the
    short, general legacy plan.

    Also returns the exercise-safety verdict and the athlete/sports screening, because this is
    the point in the flow where they belong: both are guidance about EXERCISE, and showing them
    beside the disease predictions invited them to be read as diagnoses.
    """
    report = _get_owned_report(db, report_uid, current_doctor)

    if not report.predicted_diseases:
        raise HTTPException(status_code=400,
                            detail="Run the prediction engine before generating care plans.")

    exercise_safety = report.predicted_diseases.get("exercise_safety")
    athlete_screening = report.predicted_diseases.get("athlete_screening")
    athlete_gray_zone = report.predicted_diseases.get("athlete_gray_zone")
    verdict_name = _exercise_verdict_name(report)
    age_band_label = report.predicted_diseases.get("age_band_label")

    # Exercise Safety is the hard ceiling (§4) -- Contraindicated and Indeterminate both mean no
    # plan generates for ANY disease, and the LLM is never called in either case.
    if verdict_name == "Exercise Contraindicated":
        report.care_plans = None
        report.care_plans_generated_at = datetime.utcnow()
        report.recommendation_error = None
        # No LLM call is made, so there is no run to report progress for -- but the state must
        # still land on "done", or a page polling from a previous run would spin forever.
        _set_care_plans_run(db, report, "done", 0, 0)
        return schemas.CarePlanOut(
            report_uid=report.report_uid,
            care_plans=[],
            exercise_blocked=True,
            blocked_reason=EXERCISE_BLOCKED_MESSAGE,
            athlete_screening=athlete_screening,
            exercise_safety=exercise_safety,
            athlete_gray_zone=athlete_gray_zone,
            generated_at=report.care_plans_generated_at.isoformat(),
        )

    if verdict_name == "Exercise Safety Indeterminate":
        report.care_plans = None
        report.care_plans_generated_at = datetime.utcnow()
        report.recommendation_error = None
        _set_care_plans_run(db, report, "done", 0, 0)
        return schemas.CarePlanOut(
            report_uid=report.report_uid,
            care_plans=[],
            no_disease=False,
            notice=EXERCISE_INDETERMINATE_MESSAGE,
            athlete_screening=athlete_screening,
            exercise_safety=exercise_safety,
            athlete_gray_zone=athlete_gray_zone,
            generated_at=report.care_plans_generated_at.isoformat(),
        )

    predicted = [d for d in (report.predicted_diseases.get("diseases") or [])
                if d.get("cardiac_disease_name") or d.get("name")]

    if not predicted:
        # NOT AN ERROR. The rule engine finding nothing abnormal is a normal -- and good --
        # outcome, so this returns 200 with an explanation rather than a 4xx. As a failure it
        # rendered red, next to a Retry button that could never succeed, which reads as "the
        # system broke" on precisely the report where nothing is wrong with the patient.
        report.care_plans = None
        report.care_plans_generated_at = datetime.utcnow()
        report.recommendation_error = None
        _set_care_plans_run(db, report, "done", 0, 0)

        # §5 -- borderline parameter guidance only when Exercise Safety is fully cleared AND at
        # least one parameter is borderline; a third output state, distinct from the default
        # general guidance below and from a disease-specific plan.
        borderline = report.predicted_diseases.get("borderline_parameters") or []
        if verdict_name == "No Exercise Contraindication Found" and borderline:
            return schemas.CarePlanOut(
                report_uid=report.report_uid,
                care_plans=[],
                no_disease=True,
                notice=BORDERLINE_NOTICE,
                borderline_guidance=borderline,
                athlete_screening=athlete_screening,
                exercise_safety=exercise_safety,
                athlete_gray_zone=athlete_gray_zone,
                generated_at=report.care_plans_generated_at.isoformat(),
            )

        return schemas.CarePlanOut(
            report_uid=report.report_uid,
            care_plans=[],
            no_disease=True,
            notice=NO_DISEASE_MESSAGE,
            general_guidance=general_population_guidance(age_band_label),
            athlete_screening=athlete_screening,
            exercise_safety=exercise_safety,
            athlete_gray_zone=athlete_gray_zone,
            generated_at=report.care_plans_generated_at.isoformat(),
        )

    # verdict_name is either "No Exercise Contraindication Found" (generate normally) or
    # "Exercise Restricted / Supervised Only" (every plan capped -- enforced in the prompt via
    # `exercise_safety_verdict` below, since the model must weigh it per-disease, not just once).
    patient_context = {
        "patient_age": report.patient_age,
        "patient_gender": report.patient_gender,
        "is_athlete": bool(report.is_athlete),
        "age_band": age_band_label,
        "risk_level": report.risk_level,
        "ef": getattr(report, "ef", None),
        "ivsd": getattr(report, "ivsd", None),
        "pwd": getattr(report, "pwd", None),
        "pasp": getattr(report, "pasp", None),
        "lvot_peak_gradient": getattr(report, "lvot_peak_gradient", None),
        "av_peak_gradient": getattr(report, "av_peak_gradient", None),
        "exercise_safety_verdict": {
            "name": verdict_name,
            "severity": (exercise_safety or {}).get("severity"),
            "recommendation": (exercise_safety or {}).get("recommendation"),
        },
        "compulsory_findings": _compulsory_findings(report),
    }

    meta = report.extraction_meta or {}
    for f_key in ("ivsd", "pwd"):
        f_meta = meta.get(f_key) or {}
        if isinstance(f_meta, dict):
            if f_meta.get("segment_values_mm", {}).get("max"):
                patient_context[f"{f_key}_max"] = f_meta["segment_values_mm"]["max"]
            elif f_meta.get(f"{f_key}_max"):
                patient_context[f"{f_key}_max"] = f_meta[f"{f_key}_max"]

    _set_care_plans_run(db, report, "running", 0, 1, "Unified Exercise Prescription")

    try:
        unified_markdown = generate_unified_rehab_plan(
            patient_context=patient_context,
            predicted_conditions=predicted,
            safety_tier=exercise_safety or {"tier_name": verdict_name},
            mode=mode,
        )
    except Exception as exc:
        logger.error(f"Error during unified rehab generation: {exc}")
        from app.predictor.rehab_generator import _build_deterministic_fallback, route_conditions, _resolve_safety_tier
        primary_c, secondary_c = route_conditions(predicted)
        tier_obj = _resolve_safety_tier(exercise_safety or {"tier_name": verdict_name}, patient_context, primary_c)
        unified_markdown = _build_deterministic_fallback(patient_context, primary_c, secondary_c, tier_obj)

    plans = [{
        "cardiac_disease_name": "Unified Cardiac Rehabilitation & Exercise Plan",
        "rehabilitation_exercise": unified_markdown,
    }]

    report.care_plans = plans
    report.care_plans_generated_at = datetime.utcnow()
    report.recommendation_error = None
    _set_care_plans_run(db, report, "done", len(plans), len(plans))

    return schemas.CarePlanOut(
        report_uid=report.report_uid,
        care_plans=[schemas.CarePlanItem(**p) for p in plans],
        athlete_screening=athlete_screening,
        exercise_safety=exercise_safety,
        athlete_gray_zone=athlete_gray_zone,
        combined_guidance=None,
        generated_at=report.care_plans_generated_at.isoformat(),
    )
