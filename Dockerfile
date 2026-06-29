FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY runeextract/ runeextract/
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash app

WORKDIR /app
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl[ocr,ai,rag,vector-stores]

# Allow app user to write cache dirs
RUN mkdir -p /app/.runeextract_cache /app/chroma_db && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["python", "-c", "from runeextract import run_server; run_server(host='0.0.0.0', port=8000)"]
