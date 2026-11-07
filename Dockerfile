# Scoring API image. Build from the repo root after exporting a model to deploy/model:
#   py -m creditrisk.export --name creditrisk-full-scorecard --alias production --out deploy/model
#   docker build -t credit-risk-api .
#   docker run --rm -p 8000:8000 credit-risk-api
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

COPY deploy/model ./model

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV MODEL_DIR=/app/model \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-m", "creditrisk.healthcheck"]

CMD ["python", "-m", "creditrisk.api"]
