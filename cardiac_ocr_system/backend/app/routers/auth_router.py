from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app import models, schemas
from app.auth import hash_password, verify_password, create_access_token, get_current_doctor

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/signup", response_model=schemas.Token, status_code=status.HTTP_201_CREATED)
def signup(payload: schemas.DoctorSignup, db: Session = Depends(get_db)):
    existing = db.query(models.Doctor).filter(models.Doctor.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    doctor = models.Doctor(
        full_name=payload.full_name,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        specialization=payload.specialization or "Cardiology",
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    token = create_access_token({"sub": str(doctor.id)})
    return schemas.Token(access_token=token, doctor=schemas.DoctorOut.model_validate(doctor))


@router.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    doctor = db.query(models.Doctor).filter(models.Doctor.email == form_data.username).first()
    if not doctor or not verify_password(form_data.password, doctor.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    token = create_access_token({"sub": str(doctor.id)})
    return schemas.Token(access_token=token, doctor=schemas.DoctorOut.model_validate(doctor))


@router.get("/me", response_model=schemas.DoctorOut)
def me(current_doctor: models.Doctor = Depends(get_current_doctor)):
    return current_doctor
