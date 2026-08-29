"""
Central configuration for the Cardiac OCR system.
Reads from environment variables where sensible, with safe local defaults.
"""
import os
from pathlib import Path

# PaddleOCR's bundled *_pb2.py files were generated with protoc < 3.19. When protobuf's C++/upb
# descriptor implementation is active, importing them raises "Descriptors cannot be created
# directly" and PaddleOCR becomes unusable -- which silently costs the pipeline its entire
# bounding-box geometry source. Forcing the pure-Python implementation is protobuf's own
# documented workaround. It MUST be set before anything imports google.protobuf, which is why it
# lives at the top of config.py: config is the first app module every entry point imports.
# setdefault, so an explicit environment setting still wins.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from dotenv import load_dotenv  # noqa: E402 -- must follow the protobuf guard above

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(BASE_DIR / ".env")  # secrets (e.g. GROQ_API_KEY) live in backend/.env, gitignored

# --- Database ---
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'cardiac.db'}")

# --- Auth / JWT ---
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION_1234567890")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 12  # 12 hours

# --- File uploads ---
UPLOAD_DIR = BASE_DIR / "app" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MAX_UPLOAD_SIZE_MB = 25

# --- Per-report JSON export (app/ocr/export.py) ---
# One JSON file per processed report -- every parameter with its value, confidence, source
# snippet and flag state, plus the raw OCR text. Kept OUTSIDE app/uploads so the extraction
# records are not mixed in with the original files they were produced from.
# NOTE: these files contain patient identifiers. They stay on this machine (unlike the Groq
# calls, which are redacted); treat the directory as clinical data and keep it out of git.
EXTRACTION_JSON_DIR = Path(os.getenv("EXTRACTION_JSON_DIR", str(BASE_DIR / "extractions")))

# --- OCR / Extraction ---
# Tesseract path override (Windows users often need this, e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# Confidence below this threshold => flagged for manual doctor confirmation
OCR_CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "70"))

# Fuzzy label-matching threshold (0-100). Below this, a label is not considered a match.
LABEL_MATCH_THRESHOLD = int(os.getenv("LABEL_MATCH_THRESHOLD", "82"))

# --- Document AI (Google Cloud) ---
# GCP_PROJECT_ID: GCP Console > top nav project selector, or `gcloud config get-value project`
# GCP_LOCATION: the region the Document AI processor was created in (e.g. "us", "eu") --
#               GCP Console > Document AI > Processors > (your processor) > region shown in URL
# DOCAI_PROCESSOR_ID: GCP Console > Document AI > Processors > (your processor) > Processor ID
# DOCAI_PROCESSOR_TYPE: "FORM_PARSER" (generic, works out of the box) or "CUSTOM_EXTRACTOR"
#                        (only if a Custom Extractor has been trained for this processor)
# GOOGLE_APPLICATION_CREDENTIALS: path to a service-account JSON key with Document AI API
#               access (GCP Console > IAM & Admin > Service Accounts > Keys). Read directly by
#               google-auth from os.environ -- no local constant needed for it here.
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us")
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID", "")
DOCAI_PROCESSOR_TYPE = os.getenv("DOCAI_PROCESSOR_TYPE", "FORM_PARSER")
DOCAI_TIMEOUT_SECONDS = float(os.getenv("DOCAI_TIMEOUT_SECONDS", "60"))

# --- Groq VISION (DORMANT) ---
# No longer used by the live pipeline. Structure now comes from PaddleOCR bounding-box geometry
# (app/ocr/text_extraction.py) and Groq is called TEXT-ONLY -- see GROQ_SEMANTIC_* below. The
# vision functions in groq_extractor.py and these settings are kept for the record and for any
# future opt-in path; nothing in app/ocr/pipeline.py references them.
# Reuses GROQ_API_KEY / GROQ_API_URL defined further below.
# NOTE: as of this build, "qwen/qwen3.6-27b" is the ONLY multimodal model available on this
# Groq account -- openai/gpt-oss-120b/ groq-compound / gpt-oss-* all reject image input
# with "messages[0].content must be a string". Verify before changing this default.
# This model emits <think>...</think> before its answer; groq_extractor.py strips it.
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
GROQ_VISION_TIMEOUT_SECONDS = float(os.getenv("GROQ_VISION_TIMEOUT_SECONDS", "180"))

# Legacy engine-preference switch. No longer read by app/ocr/pipeline.py, which now always runs
# the local PaddleOCR-geometry path (falling back to Tesseract flat lines). Kept so an existing
# backend/.env carrying OCR_ENGINE does not break on import.
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto").strip().lower()

# Groq returns NO per-field confidence, so confidence for Groq-sourced values is derived from
# whether an independent OCR read (PaddleOCR) corroborates the value -- never fabricated.
# extraction_meta records confidence_basis="cross_validation" so this is never mistaken for
# a real OCR confidence score.
GROQ_CORROBORATED_CONFIDENCE = float(os.getenv("GROQ_CORROBORATED_CONFIDENCE", "95"))
GROQ_UNCORROBORATED_CONFIDENCE = float(os.getenv("GROQ_UNCORROBORATED_CONFIDENCE", "60"))

# --- OCR confidence thresholds used by the structure-aware extractor ---
# SCALE: 0-100 everywhere. PaddleOCR natively returns 0-1 per line; app/ocr/text_extraction.py
# multiplies by 100 at the single point where raw Paddle output is normalized, so every
# threshold, ExtractedField.confidence and report.confidence_scores value in this codebase is
# on the same 0-100 scale. Do not mix scales.
#   Below REJECT: treated as unreadable/blank -- the line never even reaches clustering, and no
#                 value is ever extracted from it (this is what stops a blank cell from being
#                 turned into a hallucinated value downstream).
#   Between REJECT and FLAG: extracted normally but flagged for doctor review.
#   Above FLAG: extracted, no confidence-based flag (still subject to range checks).
OCR_CONFIDENCE_FLAG_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_FLAG_THRESHOLD", "80"))
OCR_CONFIDENCE_REJECT_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_REJECT_THRESHOLD", "35"))

# --- PaddleOCR detector/recogniser tuning (app/ocr/text_extraction.py) ---
# MEASURED, AND MOSTLY NOT WORTH IT. Recorded here so nobody re-runs the experiment.
#
# Every real upload is under 1085px on the long side (the worst is a 400x452 phone photo), and at
# that size the recogniser mis-reads whole labels -- the Chong Hua report returned "PSAI" for
# "IVSd". Upscaling before detection genuinely fixes that: "IVSd" and "Mild MVP(AML)" both came
# back correctly at 1600px. But scored over the whole labelled corpus it LOSES:
#
#     no upscale (0)   141 correct / 2 wrong / 0 false-positive   <-- default
#     upscale 1200     136 correct / 4 wrong / 1 false-positive
#     upscale 1600     137 correct / 4 wrong / 0 false-positive
#
# A larger page yields more, differently-shaped detection boxes, which perturbs the row/column
# clustering the whole geometry stage depends on. It fixed IVSd's label and then paired it with
# LVESd's value. The recognition win is real but smaller than the structural loss, so upscaling
# is OFF by default and left configurable for a deployment whose scans differ.
#
# If you re-enable it, RAISE OCR_DET_LIMIT_SIDE_LEN to match: PaddleOCR resizes any page whose
# long side exceeds it (limit_type="max"), so upscaling alone is silently undone. An earlier
# 1x/2x/3x experiment appeared to show "upscaling does nothing" purely because all three runs
# were resized back down to 960.
OCR_MIN_LONG_SIDE = int(os.getenv("OCR_MIN_LONG_SIDE", "0"))          # 0 = never upscale
OCR_DET_LIMIT_SIDE_LEN = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "960"))

# Dilation applied to each detected text box (PaddleOCR default 1.5). Lowering it to 1.2 to stop
# adjacent columns fusing ("LastIVSTd", "RAD (major3.3") scored the same 141 correct but turned
# one miss into a WRONG value, so the default stands -- a wrong number costs more than a miss.
OCR_DET_UNCLIP_RATIO = float(os.getenv("OCR_DET_UNCLIP_RATIO", "1.5"))

# PaddleOCR silently DISCARDS any recognition below this score (its default is 0.5) before this
# codebase ever sees it. 0.5 on Paddle's scale is 50 on ours, ABOVE
# OCR_CONFIDENCE_REJECT_THRESHOLD (35) -- so Paddle's cutoff was pre-empting the threshold this
# project documents as authoritative, and a marginal read vanished with no flag and no reason
# instead of surfacing for doctor review.
# Scored exactly neutral (141 correct either way); kept at 0.3 for that design reason, not for
# an accuracy gain, so that our own reject/flag logic is the one that actually governs.
OCR_DROP_SCORE = float(os.getenv("OCR_DROP_SCORE", "0.3"))

# --- Geometric table reconstruction (app/ocr/text_extraction.py) ---
# Two OCR boxes belong to the same ROW when their vertical [top, top+height] spans overlap by
# more than this fraction of the SHORTER box's height.
OCR_ROW_OVERLAP_RATIO = float(os.getenv("OCR_ROW_OVERLAP_RATIO", "0.5"))
# A gap between consecutive sorted left-edges larger than this STARTS A NEW COLUMN.
#
# Expressed as a multiple of the page's own median OCR line height, NOT in absolute pixels,
# because the threshold has to hold at every input resolution. The old fixed 40px default was
# calibrated for 300-DPI rasterized A4 (~2480px wide, median line height ~40px). Fed a 630px-wide
# phone photo -- median line height 14px, real inter-column gaps 16-27px -- every one of those
# gaps fell under 40 and the columns silently merged into one cell, so "9.23 cm2" and "IVSTd"
# ended up in the same grid cell and the parameter was lost. Scaling with text height reproduces
# the old behaviour at 300 DPI while also working on a low-resolution upload.
#
# Measured on that page the gap distribution splits cleanly: >=16px are real column boundaries,
# <=10px are within-column jitter, so the usable band is 1.1x-1.9x the median line height.
# 1.5 was chosen by sweeping the ratio against tests/fixtures/ground_truth.json: 1.0 and 1.2
# over-split (a row gained enough cells that a label paired with a neighbouring column's value,
# costing LA_Diameter and LA_Diameter_Indexed), while 1.5 and 1.8 both scored identically and
# best. 1.5 is the midpoint of that stable band rather than its edge.
OCR_COLUMN_GAP_RATIO = float(os.getenv("OCR_COLUMN_GAP_RATIO", "1.5"))

# Explicit pixel override. Set OCR_COLUMN_GAP_PX in the environment to pin an absolute gap for a
# known fixed-resolution template and bypass the ratio above; left unset, the ratio wins. Also
# used as the fallback when a page yields no usable line heights.
OCR_COLUMN_GAP_PX = float(os.getenv("OCR_COLUMN_GAP_PX", "40"))
OCR_COLUMN_GAP_PX_EXPLICIT = "OCR_COLUMN_GAP_PX" in os.environ

# Semantic (text-only) Groq model used by app/ocr/extractor.py to resolve label variants and
# read qualitative findings out of narrative prose. NEVER receives the page image -- Stage 1
# already solved structure; this model only does language understanding.
GROQ_SEMANTIC_MODEL = os.getenv("GROQ_SEMANTIC_MODEL", "openai/gpt-oss-120b")
GROQ_SEMANTIC_TIMEOUT_SECONDS = float(os.getenv("GROQ_SEMANTIC_TIMEOUT_SECONDS", "45"))
# Set to "0"/"false" to run fully offline/deterministic (dictionary matching only, no API call).
GROQ_SEMANTIC_ENABLED = os.getenv("GROQ_SEMANTIC_ENABLED", "1").strip().lower() not in ("0", "false", "no")

# --- Groq LLM (rehab exercise + diet plan generation) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))

