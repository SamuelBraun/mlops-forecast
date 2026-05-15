# Streamlit dashboard
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY streamlit_app/ streamlit_app/

RUN pip install --upgrade pip && pip install -e "."

ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="http://api:8000"
ENV MLFLOW_TRACKING_URI="http://mlflow:5000"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
