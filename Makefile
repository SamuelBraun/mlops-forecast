.PHONY: install install-dev lint typecheck test run download-data serve dashboard clean help

PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
# Kedro 0.19 warns about Python 3.13 at import time. The warning fires before
# pytest/kedro CLI filters take effect, so we silence it via env var.
KEDRO_ENV := KEDRO_DISABLE_TELEMETRY=1 PYTHONWARNINGS="ignore::kedro.KedroPythonVersionWarning"
KEDRO  := $(KEDRO_ENV) $(VENV)/bin/kedro
PYTEST := $(KEDRO_ENV) $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
MYPY   := $(VENV)/bin/mypy

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install: $(VENV)/bin/activate  ## Create venv and install production deps
	$(PIP) install -e ".[dev]"
	@echo "✓ Environment ready. Activate with: source $(VENV)/bin/activate"

install-dev: install  ## Install dev extras + pre-commit hooks
	$(VENV)/bin/pre-commit install

lint:  ## Run ruff linter and formatter check
	$(RUFF) check src/ tests/ api/ streamlit_app/
	$(RUFF) format --check src/ tests/ api/ streamlit_app/

lint-fix:  ## Auto-fix lint issues
	$(RUFF) check --fix src/ tests/ api/ streamlit_app/
	$(RUFF) format src/ tests/ api/ streamlit_app/

typecheck:  ## Run mypy type checker
	$(MYPY) src/mlops_forecast/

test:  ## Run pytest suite with coverage
	$(PYTEST) tests/ -v

test-fast:  ## Run tests without slow integration tests
	$(PYTEST) tests/ -v -m "not slow"

download-data:  ## Fetch OPSD dataset into data/01_raw/
	$(PYTHON) scripts/download_data.py

run:  ## Run full Kedro pipeline end-to-end
	$(KEDRO) run --pipeline __default__

run-quality:  ## Run data_quality pipeline only
	$(KEDRO) run --pipeline data_quality

run-cleaning:  ## Run data_cleaning pipeline only
	$(KEDRO) run --pipeline data_cleaning

run-features:  ## Run data_feat_engineering pipeline only
	$(KEDRO) run --pipeline data_feat_engineering

run-split:  ## Run data_split pipeline only
	$(KEDRO) run --pipeline data_split

run-train:  ## Run model_train pipeline only
	$(KEDRO) run --pipeline model_train

run-select:  ## Run model_selection pipeline only
	$(KEDRO) run --pipeline model_selection

run-predict:  ## Run model_predict pipeline only
	$(KEDRO) run --pipeline model_predict

run-drift:  ## Run data_drifts pipeline only
	$(KEDRO) run --pipeline data_drifts

feast-publish:  ## Publish Kedro features to Feast (apply + materialize)
	$(VENV)/bin/python scripts/publish_to_feast.py

feast-demo:  ## Smoke test Feast offline + online retrieval
	$(VENV)/bin/python scripts/feast_demo.py

report-figures:  ## Regenerate PNGs in report/figures/
	$(VENV)/bin/python scripts/generate_report_figures.py
	cp data/08_reporting/explainability/shap_summary.png report/figures/shap_summary.png

report:  ## Build the LaTeX report into report/MLops_Report.pdf
	cd report && tectonic MLops_Report.tex

demo:  ## Full end-to-end: fetch data, run pipelines, publish Feast, bring services up
	@if [ ! -f data/01_raw/opsd_germany_hourly.csv ]; then \
		echo "==> Downloading OPSD dataset"; \
		$(VENV)/bin/python scripts/download_data.py; \
	else \
		echo "==> OPSD dataset already present, skipping download"; \
	fi
	@echo "==> Running all eight Kedro pipelines"
	$(KEDRO) run --pipeline __default__
	@echo "==> Publishing features into Feast"
	$(VENV)/bin/python scripts/publish_to_feast.py
	@echo "==> Bringing up MLflow + FastAPI + Streamlit (docker compose)"
	docker compose up --build -d
	@echo ""
	@echo "Demo ready:"
	@echo "  MLflow UI:    http://localhost:5000"
	@echo "  FastAPI docs: http://localhost:8000/docs"
	@echo "  Streamlit:    http://localhost:8501"

serve:  ## Start MLflow + FastAPI + Streamlit via docker-compose
	docker compose up --build -d
	@echo "MLflow UI:    http://localhost:5000"
	@echo "FastAPI docs: http://localhost:8000/docs"
	@echo "Streamlit:    http://localhost:8501"

serve-down:  ## Stop all services
	docker compose down

dashboard:  ## Open Streamlit dashboard in browser (requires services running)
	open http://localhost:8501 || xdg-open http://localhost:8501

mlflow-ui:  ## Start local MLflow UI (no Docker)
	$(VENV)/bin/mlflow ui --backend-store-uri ./mlruns --port 5000

viz:  ## Open Kedro pipeline visualisation
	$(KEDRO) viz

clean:  ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
	@echo "✓ Cleaned build artefacts (data/ and mlruns/ untouched)"

clean-data:  ## Remove generated data layers (preserves raw + gitkeeps)
	find data/02_intermediate data/03_primary data/04_feature \
	     data/05_model_input data/06_models data/07_model_output data/08_reporting \
	     -type f ! -name ".gitkeep" -delete
	@echo "✓ Intermediate data layers cleared"

.DEFAULT_GOAL := help
