import uuid
from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import relationship

from app.database import Base


def gen_uuid() -> str:
    return uuid.uuid4().hex


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(120), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    specialization = Column(String(120), default="Cardiology")
    created_at = Column(DateTime, default=datetime.utcnow)

    reports = relationship("CardiacReport", back_populates="doctor")


class CardiacReport(Base):
    """
    One row per uploaded 2D Echo report.
    Holds the unique patient/report ID, patient name, every extracted cardiac
    parameter as its own column (exact value as read from the report), plus
    supporting metadata (confidence scores, flags, raw OCR text) as JSON.
    """
    __tablename__ = "cardiac_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_uid = Column(String(40), unique=True, index=True, default=gen_uuid)  # unique patient/report ID

    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    patient_name = Column(String(150), nullable=False)
    patient_age = Column(String(10), nullable=True)
    patient_gender = Column(String(10), nullable=True)
    # Outdoor Sports Person flag (§6). Combined with age + gender, selects which reference table
    # ivsd/pwd/lvidd/lv_mass/rwt/ra_size/rv_size/e_a_ratio resolve against. Persists with the
    # record exactly like age/gender, and is threaded into every rule function that needs it.
    is_athlete = Column(Boolean, default=False, nullable=False)

    original_filename = Column(String(255))
    status = Column(String(30), default="uploaded")  # uploaded -> processing -> extracted -> reviewed
    progress = Column(Integer, default=0)             # 0-100 for the progress bar
    progress_stage = Column(String(80), default="Queued")

    # WHY a failed report failed, as a stable machine value (see pipeline.FAILURE_* constants):
    # wrong_file | no_text | no_parameters | error. progress_stage carries the sentence a human
    # reads; this carries the cause the UI branches on -- so "you uploaded the wrong document"
    # can be presented differently from "we could not read your report", without the frontend
    # pattern-matching English prose that is free to be reworded.
    failure_reason = Column(String(40), nullable=True)

    # ---- Extracted cardiac parameters (exact values as found in report) ----
    # FLEXIBLE TYPING BY DESIGN: every parameter column is TEXT, never INT/FLOAT.
    # A 2D-Echo report prints EITHER a measurement ("40%", "13 mm") OR a descriptive finding
    # ("Normal", "Enlarged", "Mild Regurgitation") in the very same field, and both are equally
    # valid clinical readings. A numeric column would reject half of the real reports outright,
    # and a length-limited VARCHAR(20) cannot hold "Mild Regurgitation" -- so every one of these
    # is Text. Which of the two a stored value is, is recorded as value_type in extraction_meta
    # (see below); app/predictor/normalize.py parses these strings back into typed values and
    # already returns None for anything non-numeric, so descriptive values pass through it
    # safely rather than raising a coercion error.
    ef = Column(Text)              # Ejection Fraction, e.g. "58%"
    lvidd = Column(Text)           # LV Internal Diameter, diastole
    lvids = Column(Text)           # LV Internal Diameter, systole
    ivsd = Column(Text)            # Interventricular Septum thickness, diastole
    pwd = Column(Text)             # Posterior Wall thickness, diastole
    la_diameter = Column(Text)     # Left Atrium diameter
    ao_diameter = Column(Text)     # Aortic diameter (generic, landmark unspecified)
    rv_size = Column(Text)         # Right Ventricle size
    mv_finding = Column(Text)      # Mitral Valve finding
    av_finding = Column(Text)      # Aortic Valve finding
    tv_finding = Column(Text)      # Tricuspid Valve finding
    pv_finding = Column(Text)      # Pulmonary Valve finding
    e_a_ratio = Column(Text)       # E/A ratio
    pasp = Column(Text)            # Pulmonary Artery Systolic Pressure
    wall_motion = Column(Text)     # Wall motion abnormality findings (free text)

    # ---- New parameters (Phase 2) ----
    ivss = Column(Text)                  # Interventricular Septum thickness, systole
    pws = Column(Text)                   # Posterior Wall thickness, systole
    ra_size = Column(Text)               # Right Atrium size (numeric or qualitative)
    ivc = Column(Text)                   # Inferior Vena Cava diameter
    pericardial_effusion = Column(Text)  # Pericardial effusion finding
    clots_thrombus = Column(Text)        # Clots / intracardiac thrombus finding

    # ---- New parameters (Phase 3: structure-aware extraction) ----
    ao_annulus = Column(Text)            # Aortic annulus diameter
    ao_root = Column(Text)               # Aortic root / sinus of Valsalva diameter
    ao_stj = Column(Text)                # Sinotubular junction diameter
    lvidd_indexed_value = Column(Text)   # LVIDd / BSA (cm/m^2) -- distinct from lvidd
    la_diameter_indexed_value = Column(Text)  # LA diameter / BSA (cm/m^2) -- distinct from la_diameter
    bsa = Column(Text)                   # BSA in m^2 (from report, calculated, or manual)
    height = Column(Text)                # Height in cm
    weight = Column(Text)                # Weight in kg
    bsa_source = Column(Text)            # "from report", "calculated", "manual entry", or "missing"


    # ---- Valve Doppler (Phase 4) ----
    # Peak velocity and peak gradient are DISTINCT from the *_finding columns above. Those hold a
    # qualitative finding ("Mild Regurgitation"); these hold the numbers the report prints for the
    # same valve. A report can carry either, both, or neither -- Hi-Precision prints velocity and
    # gradient for all four valves with the regurgitation columns blank.
    av_peak_velocity = Column(Text)      # Aortic valve peak velocity (m/s)
    av_peak_gradient = Column(Text)      # Aortic valve peak gradient (mmHg)
    av_area = Column(Text)               # Aortic valve area (cm2)
    mv_peak_velocity = Column(Text)      # Mitral inflow peak (E-wave) velocity (m/s)
    mv_peak_gradient = Column(Text)      # Mitral valve peak gradient (mmHg)
    tv_peak_velocity = Column(Text)      # Tricuspid regurgitant jet velocity (m/s) -- PASP source
    tv_peak_gradient = Column(Text)      # Tricuspid peak gradient (mmHg)
    pv_peak_velocity = Column(Text)      # Pulmonic valve peak velocity (m/s)
    pv_peak_gradient = Column(Text)      # Pulmonic valve peak gradient (mmHg)
    lvot_peak_velocity = Column(Text)    # LVOT peak velocity (m/s)
    lvot_peak_gradient = Column(Text)    # LVOT peak gradient (mmHg)
    lvot_vti = Column(Text)              # LVOT velocity-time integral (cm)
    mv_area = Column(Text)               # Mitral valve area (cm2) -- mitral stenosis grading
    mv_annulus = Column(Text)            # Mitral annulus diameter (cm)
    tv_annulus = Column(Text)            # Tricuspid annulus diameter (cm)
    pv_annulus = Column(Text)            # Pulmonic annulus diameter (cm)

    # ---- Wall condition (Phase 5, WALL_PARAMETER_DICTIONARY) ----
    relative_wall_thickness = Column(Text)  # RWT, unitless ratio (2*PWd/LVIDd)
    lv_mass = Column(Text)                  # LV mass OR LV mass index -- see the note on the
                                            # LV_Mass entry in parameter_dict.py: the supplied
                                            # dictionary merges absolute and BSA-indexed mass,
                                            # which are clinically distinct.
    rwma = Column(Text)                     # Regional wall motion abnormality (qualitative)
    septal_wall_motion = Column(Text)
    anterior_wall_motion = Column(Text)
    inferior_wall_motion = Column(Text)
    lateral_wall_motion = Column(Text)
    posterior_wall_motion = Column(Text)
    apical_wall_motion = Column(Text)

    # ---- Impression / Conclusion ----
    # The report's own concluding prose, stored VERBATIM. Never summarized, reworded or
    # reformatted: this is the cardiologist's own wording and the review page shows it as-is.
    # Deliberately separate from the parameter columns -- it is evidence, not an extracted value.
    impression_text = Column(Text)

    # ---- Supporting / audit data ----
    raw_ocr_text = Column(Text)          # UNFILTERED OCR text -- every line the engine read,
                                          # with no confidence filtering, for the verification
                                          # preview page. Deliberately not the filtered text the
                                          # extractor worked from: a doctor checking why a field
                                          # came out blank needs to see the unreadable region.
    confidence_scores = Column(JSON)     # {"ef": 91.2, "lvidd": 60.0, ...} -- 0-100 scale
    flagged_params = Column(JSON)        # ["lvidd", "pasp"] -> needs doctor confirmation
    doctor_confirmed = Column(JSON)      # params the doctor has manually confirmed/edited
    extra_findings = Column(Text)        # any additional free text notes
    extraction_meta = Column(JSON)       # Per-field audit trail, keyed by db_field:
                                          #   value_type        "numeric" | "qualitative" | None
                                          #   source            "table" | "narrative" |
                                          #                     "form_field" | "flat_line_fallback"
                                          #   confidence        0-100
                                          #   final_stored_value / source_snippet
                                          #   flagged / flag_reason
                                          # NUMERIC FIELDS ONLY additionally carry
                                          #   raw_detected_value / raw_detected_unit /
                                          #   conversion_applied
                                          # -- qualitative findings omit those three keys
                                          # entirely, because unit and conversion have no
                                          # meaning for a word like "Normal".
                                          # Plus "_pipeline": {...} for engine/quality info.

    # ---- Disease prediction (rule engine) ----
    predicted_diseases = Column(JSON)      # full evaluate_all() result: {diseases, normal_heart, athlete_screening, risk_summary, ...}
    risk_level = Column(String(20))        # "Low" | "Moderate" | "High" | "Normal"
    rules_evaluated = Column(Integer)      # how many of the 36 rules had enough data to evaluate
    prediction_generated_at = Column(DateTime)

    # ---- Groq LLM rehab/diet recommendation ----
    rehab_plan = Column(Text)              # generated rehabilitation/exercise plan
    diet_plan = Column(Text)               # generated diet plan
    recommendation_error = Column(Text)    # last LLM call error, if any
    recommendation_generated_at = Column(DateTime)

    # ---- Per-disease care plans (v4.0) ----
    # [{"cardiac_disease_name", "rehabilitation_exercise"}, ...] -- one entry per predicted
    # disease. Exercise only: this flow no longer generates dietary guidance, though rows written
    # before that change may still carry a "recommended_diet" key, which is simply not read.
    # Deliberately a NEW column rather than a reuse of rehab_plan/diet_plan above: those hold a
    # single markdown plan for the whole report, and plans already generated for existing reports
    # must stay readable.
    care_plans = Column(JSON)
    care_plans_generated_at = Column(DateTime)

    # Live progress of a care-plan run, polled by the report page while the run is in flight:
    #   {"status": idle|running|done|error, "done": int, "total": int,
    #    "progress": 0-100, "current": "<disease being written>", "error": str|None}
    # MEASURED, NOT ESTIMATED. `done`/`total` are real counts of completed Groq calls, which is
    # the whole reason generate_disease_care_plans() makes one call per disease. Nothing here is
    # ever derived from a timer -- a percentage that advances on a clock while the server is stuck
    # is worse than no percentage, because it looks like information.
    # One JSON column rather than five scalar ones: this is a single short-lived run state that is
    # always read and written as a unit, and it is exactly the shape the progress endpoint returns.
    care_plans_run = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    doctor = relationship("Doctor", back_populates="reports")
    confirmations = relationship("PredictionConfirmation", back_populates="report",
                                  cascade="all, delete-orphan")

    PARAM_FIELDS = [
        "ef", "lvidd", "lvids", "ivsd", "pwd", "la_diameter", "ao_diameter",
        "rv_size", "mv_finding", "av_finding", "tv_finding", "pv_finding",
        "e_a_ratio", "pasp", "wall_motion",
        "ivss", "pws", "ra_size", "ivc", "pericardial_effusion", "clots_thrombus",
        "ao_annulus", "ao_root", "ao_stj", "lvidd_indexed_value", "la_diameter_indexed_value",
        "av_peak_velocity", "av_peak_gradient", "av_area", "mv_peak_velocity", "mv_peak_gradient",
        "tv_peak_velocity", "tv_peak_gradient", "pv_peak_velocity", "pv_peak_gradient",
        "lvot_peak_velocity", "lvot_peak_gradient", "lvot_vti", "mv_area",
        "mv_annulus", "tv_annulus", "pv_annulus",
        "relative_wall_thickness", "lv_mass", "rwma", "septal_wall_motion",
        "anterior_wall_motion", "inferior_wall_motion", "lateral_wall_motion",
        "posterior_wall_motion", "apical_wall_motion",
        "bsa", "height", "weight",
    ]

    def to_dict(self):
        return {
            "id": self.id,
            "report_uid": self.report_uid,
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "is_athlete": bool(self.is_athlete),
            "bsa_source": self.bsa_source,
            "original_filename": self.original_filename,

            "status": self.status,
            "progress": self.progress,
            "progress_stage": self.progress_stage,
            "failure_reason": self.failure_reason,
            "parameters": {f: getattr(self, f) for f in self.PARAM_FIELDS},
            "impression_text": self.impression_text,
            "confidence_scores": self.confidence_scores or {},
            "flagged_params": self.flagged_params or [],
            "doctor_confirmed": self.doctor_confirmed or [],
            "extra_findings": self.extra_findings,
            "raw_ocr_text": self.raw_ocr_text,
            "extraction_meta": self.extraction_meta or {},
            "predicted_diseases": self.predicted_diseases or None,
            "risk_level": self.risk_level,
            "rules_evaluated": self.rules_evaluated,
            "prediction_generated_at": self.prediction_generated_at.isoformat() if self.prediction_generated_at else None,
            "rehab_plan": self.rehab_plan,
            "diet_plan": self.diet_plan,
            "recommendation_error": self.recommendation_error,
            "recommendation_generated_at": self.recommendation_generated_at.isoformat() if self.recommendation_generated_at else None,
            "care_plans": self.care_plans or None,
            "care_plans_run": self.care_plans_run or None,
            "care_plans_generated_at": self.care_plans_generated_at.isoformat() if self.care_plans_generated_at else None,
            "confirmations": [c.to_dict() for c in (self.confirmations or [])],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PredictionConfirmation(Base):
    """One clinician's verdict on ONE predicted disease for ONE report.

    The feedback loop the rule engine learns from: whether the physician agrees with a
    prediction, plus free-text corrections (a disease the engine missed, or one it called wrongly).

    KEYED ON (report_id, disease_name) as specified. Disease names encode severity, so a re-run
    after a parameter correction can produce a different name -- "Moderate Mitral Regurgitation"
    becomes "Mild Mitral Regurgitation". When that happens the row is NOT deleted: it is marked
    `is_stale` and still shown. A doctor's written correction is the most valuable data in this
    system and must never disappear because a number changed.
    """
    __tablename__ = "prediction_confirmations"
    __table_args__ = (UniqueConstraint("report_id", "disease_name", name="uq_report_disease"),)

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(Integer, ForeignKey("cardiac_reports.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)

    disease_name = Column(String(200), nullable=False, index=True)
    confirmed = Column(Boolean, default=False, nullable=False)
    clinician_notes = Column(Text)

    # True when the most recent prediction run no longer produced this disease.
    is_stale = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    report = relationship("CardiacReport", back_populates="confirmations")

    def to_dict(self):
        return {
            "disease_name": self.disease_name,
            "confirmed": bool(self.confirmed),
            "clinician_notes": self.clinician_notes or "",
            "is_stale": bool(self.is_stale),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
