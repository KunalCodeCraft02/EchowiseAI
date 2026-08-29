FROM python:3.11-slim

# Native deps opencv-python-headless / paddlepaddle need on Debian slim, plus the
# Tesseract binary for the last-resort OCR fallback (pytesseract).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# protobuf==3.20.2 is pinned hard in requirements.txt for paddlepaddle compatibility --
# install exactly as pinned, do not let pip resolve it upward.
COPY cardiac_ocr_system/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cardiac_ocr_system/backend/app/ ./app/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
