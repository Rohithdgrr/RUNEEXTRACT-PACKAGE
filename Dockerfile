FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY runeextract/ runeextract/

RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000

CMD ["python", "-m", "runeextract"]
