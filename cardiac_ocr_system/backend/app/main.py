from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.database import init_db
from app.routers import auth_router, reports

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Cardiac OCR, Information Extraction & Predictive Analytics System",
    description="OCR + information extraction pipeline for 2D Echo reports.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(auth_router.router)
app.include_router(reports.router)

# ---- Serve frontend pages ----
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _page(name: str):
    return FileResponse(str(STATIC_DIR / "pages" / name))


@app.get("/")
def root():
    return _page("home.html")


@app.get("/login")
def login_page():
    return _page("login.html")


@app.get("/signup")
def signup_page():
    return _page("signup.html")


@app.get("/dashboard")
def dashboard_page():
    return _page("dashboard.html")


@app.get("/upload")
def upload_page():
    return _page("upload.html")


@app.get("/processing/{report_uid}")
def processing_page(report_uid: str):
    return _page("processing.html")


@app.get("/report/{report_uid}")
def report_page(report_uid: str):
    return _page("report.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}
