FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY config ./config
COPY src ./src
COPY models ./models
# Editable install keeps the project root at /app, where config/ and data/ live.
RUN pip install -e . && useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app/data

# Trained models ship inside the image (train locally, commit, push); raw payloads live on the /app/data volume.
ENV DOTA_MODELS_DIR=/app/models \
    DOTA_RAW_STORE_DIR=/app/data/raw

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5)"]

CMD ["dota-predictor", "serve"]
