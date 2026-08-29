from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field


# ---------- Auth ----------
class DoctorSignup(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    specialization: Optional[str] = "Cardiology"


class DoctorLogin(BaseModel):
    email: EmailStr
    password: str


class DoctorOut(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    specialization: Optional[str] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    doctor: DoctorOut


# ---------- Reports ----------
class ReportCreateResponse(BaseModel):
    report_uid: str
    id: int
    status: str
    message: str


class ProgressOut(BaseModel):
    report_uid: str
    status: str
    progress: int
    progress_stage: str
    # Only set when status == "failed". The cause the processing page branches on:
    # wrong_file | no_text | no_parameters | error (app/ocr/pipeline.py FAILURE_* constants).
    failure_reason: Optional[str] = None


class ParameterUpdate(BaseModel):
    field: str
    value: str


class ParameterSource(BaseModel):
    """How one parameter's value got into the report -- the audit trail behind a blank field.

    Written only for fields the doctor filled on the review page. An OCR-extracted value keeps
    whatever extraction_meta the pipeline wrote for it and is never touched by this.
    """
    source: str                                   # doctor_dropdown | doctor_custom
    option: Optional[str] = None                  # the dropdown option chosen
    age_group: Optional[str] = None               # which age group's table resolved it
    resolved_display: Optional[str] = None        # the published band, e.g. "< 35 %"
    unit: Optional[str] = None


class ReportUpdate(BaseModel):
    parameters: Dict[str, str] = {}
    confirmed_fields: Optional[List[str]] = None
    extra_findings: Optional[str] = None
    is_athlete: Optional[bool] = None
    # Provenance for doctor-filled fields, keyed by db field name. Optional: a client that does
    # not send it changes nothing about how values are stored.
    parameter_sources: Optional[Dict[str, ParameterSource]] = None


class ReportOut(BaseModel):
    id: int
    report_uid: str
    patient_name: str
    patient_age: Optional[str] = None
    patient_gender: Optional[str] = None
    is_athlete: bool = False
    bsa_source: Optional[str] = None
    original_filename: Optional[str] = None
    status: str
    progress: int
    progress_stage: str
    failure_reason: Optional[str] = None
    parameters: Dict[str, Any]
    confidence_scores: Dict[str, Any]
    flagged_params: List[str]
    doctor_confirmed: List[str]
    extra_findings: Optional[str] = None
    raw_ocr_text: Optional[str] = None
    impression_text: Optional[str] = None
    extraction_meta: Optional[Dict[str, Any]] = None
    predicted_diseases: Optional[Dict[str, Any]] = None
    risk_level: Optional[str] = None
    rules_evaluated: Optional[int] = None
    prediction_generated_at: Optional[str] = None
    rehab_plan: Optional[str] = None
    recommendation_error: Optional[str] = None
    recommendation_generated_at: Optional[str] = None
    care_plans: Optional[List[Dict[str, Any]]] = None
    care_plans_generated_at: Optional[str] = None
    confirmations: List[Dict[str, Any]] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ---------- Prediction / Recommendation ----------
class PredictionOut(BaseModel):
    report_uid: str
    risk_level: Optional[str] = None
    rules_total: int
    rules_evaluated: int
    normal_heart: Optional[Dict[str, Any]] = None
    diseases: List[Dict[str, Any]] = []
    athlete_screening: Optional[Dict[str, Any]] = None
    exercise_safety: Optional[Dict[str, Any]] = None
    # §6 -- unconditional Athlete's Heart vs HCM differential screen. Requires explicit doctor
    # confirmation before use elsewhere; see PredictionConfirmation for the confirm/notes loop.
    athlete_gray_zone: Optional[Dict[str, Any]] = None
    # §5 -- filled parameters mildly outside the age/sex-resolved normal band that no disease
    # rule claimed. Only populated when no disease was predicted.
    borderline_parameters: List[Dict[str, Any]] = []
    # Which age band's thresholds were applied, and how many of the nine compulsory groups the
    # report could actually be graded on. Present here as well as in the stored result so a
    # fresh run and a page reload show the same thing.
    age_band: Optional[str] = None
    age_band_label: Optional[str] = None
    # The four-band clinical age group (children/youth/adults/elderly) the value table and the
    # confidence/prediction/severity scores were resolved against.
    age_group: Optional[str] = None
    age_group_label: Optional[str] = None
    age_unknown: bool = False
    pending_pediatric_review: bool = False
    pediatric_notice: Optional[str] = None
    compulsory_coverage: Optional[Dict[str, Any]] = None
    risk_summary: Optional[Dict[str, Any]] = None
    disclaimer: str
    generated_at: Optional[str] = None


class RecommendationOut(BaseModel):
    """Exercise only. `exercise_blocked` is set when the Exercise Safety verdict is
    'Exercise Contraindicated' -- no plan is generated at all in that case."""
    report_uid: str
    rehab_plan: Optional[str] = None
    key_precautions: Optional[str] = None
    exercise_blocked: bool = False
    blocked_reason: Optional[str] = None
    error: Optional[str] = None
    generated_at: Optional[str] = None


# ---------- Clinician feedback (v4.0) ----------
class ConfirmationIn(BaseModel):
    """One physician verdict on one predicted disease."""
    disease_name: str = Field(..., min_length=1, max_length=200)
    confirmed: Optional[bool] = None
    clinician_notes: Optional[str] = None


class ConfirmationOut(BaseModel):
    disease_name: str
    confirmed: bool
    clinician_notes: str = ""
    is_stale: bool = False
    updated_at: Optional[str] = None


class CarePlanItem(BaseModel):
    """One per-disease plan. Exercise only -- this flow no longer carries dietary guidance."""
    cardiac_disease_name: str
    rehabilitation_exercise: str


class CarePlanProgressOut(BaseModel):
    """Live state of a care-plan run, polled while the generate request is still in flight.

    Every number here is measured: `done` counts Groq calls that have actually returned, and
    `progress` is derived from it. Nothing advances on a timer.
    """
    report_uid: str
    status: str                            # idle | running | done | error
    progress: int = 0                      # 0-100, = done/total
    done: int = 0
    total: int = 0
    current: Optional[str] = None          # disease currently being written
    error: Optional[str] = None


class CarePlanOut(BaseModel):
    report_uid: str
    care_plans: List[CarePlanItem] = []
    exercise_blocked: bool = False
    blocked_reason: Optional[str] = None
    # No abnormal finding matched, so no plan was generated. A RESULT, not a failure -- the
    # request succeeds and the page states the outcome rather than showing an error.
    no_disease: bool = False
    notice: Optional[str] = None
    athlete_screening: Optional[Dict[str, Any]] = None
    exercise_safety: Optional[Dict[str, Any]] = None
    athlete_gray_zone: Optional[Dict[str, Any]] = None
    # Deterministic (non-LLM) reasoning shown above the per-disease plans when 2+ diseases were
    # predicted, naming the most restrictive relevant constraint and which finding drove it (§4).
    combined_guidance: Optional[str] = None
    # §5 -- shown instead of the per-disease plans when no disease was predicted but >=1 filled
    # parameter sits in the borderline band just below a disease-triggering threshold.
    borderline_guidance: Optional[List[Dict[str, Any]]] = None
    # §4 no-disease case -- general ACSM/AHA baseline activity guidance, clearly labelled as
    # general rather than condition-specific.
    general_guidance: Optional[str] = None
    generated_at: Optional[str] = None
