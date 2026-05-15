# Multi-stage build for FastAPI serving.
#
# The runtime image only ships what the API needs at request time: the
# `serving` and `api` packages plus their direct deps (FastAPI, MLflow client,
# LightGBM, pandas). Kedro / Streamlit / Great Expectations / Evidently / SHAP
# are explicitly NOT installed — they live in the training image.

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy only what's needed to install: package metadata + the lean modules
# the API imports at request time.
COPY pyproject.toml ./
COPY src/mlops_forecast/__init__.py src/mlops_forecast/__init__.py
COPY src/mlops_forecast/serving/ src/mlops_forecast/serving/
COPY api/ api/

# Install the bare minimum runtime deps. We deliberately don't run
# `pip install -e .` — that would pull in Kedro, SHAP, Great Expectations etc.
# Versions intentionally loose; lock with `pip freeze > requirements.lock`
# in CI when stability matters more than ergonomics.
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install \
        "mlflow>=3.0,<4.0" \
        "lightgbm>=4.5,<5.0" \
        "pandas==2.2.3" \
        "numpy>=2.1" \
        "pyarrow>=17,<24" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32" \
        "pydantic>=2.9" \
        "scikit-learn>=1.5"

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Make the `mlops_forecast` package importable from /app/src
ENV PYTHONPATH="/app/src"
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"
ENV MODEL_NAME="ElectricityForecast"
ENV MODEL_STAGE="Production"

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
