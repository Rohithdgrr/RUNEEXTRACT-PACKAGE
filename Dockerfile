FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Rohithdgrr/RUNEEXTRACT-PACKAGE"
LABEL org.opencontainers.image.description="RuneExtract - One extraction API for every document type"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runeextract with all extras
COPY dist/*.whl /tmp/ 2>/dev/null || true
RUN pip install --no-cache-dir runeextract[ocr,ai] 2>/dev/null || \
    pip install --no-cache-dir runeextract

ENTRYPOINT ["runeextract"]
CMD ["--help"]
