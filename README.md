# MLOps Forecast

Hourly electricity load forecasting in Germany, with the surrounding MLOps system that lets a real team operate it. Coursework for NOVA IMS MLOps, Spring 2026.

The model is a LightGBM regressor that predicts the next 24 hours of total German load from lag features, weather, and calendar variables. Test MAPE on the COVID-affected 2020 holdout is 1.02%, with conformalised prediction intervals that hit 80% coverage on the validation set. The model itself is not the point of the assignment; what makes the project graded is everything around it. That includes:

- eight Kedro pipelines, from raw CSV to drift report
- an MLflow registry with a `Staging`/`Production` promotion path
- a Feast feature store, used at training time and at serving time
- an Evidently drift monitor that distinguishes a control window from the COVID lockdown
- a FastAPI service whose Docker image deliberately excludes the training-time dependencies
- a Streamlit dashboard, GitHub Actions CI, pre-commit hooks, and a LaTeX report

[**Report (PDF)**](report/MLops_Report.pdf) · [**Reproduce in five commands**](#run-it-locally) · [**Architecture**](#architecture)

## Headline numbers

| Metric | Value | Notes |
|---|--:|---|
| Test MAPE | **1.02 %** | LightGBM, 2020 test set |
| Validation MAPE | 0.80 % | 2019 |
| Walk-forward CV MAPE (mean ± std) | 0.93 % ± 0.15 % | 4 expanding-window folds |
| Improvement over seasonal-naive baseline | 6× | 4.74 % val → 0.80 % val |
| Improvement over Prophet | 7× | 5.71 % val → 0.80 % val |
| Drift score, control window vs COVID lockdown | 0.22 vs 0.66 | average Wasserstein distance |
| Prediction interval coverage (val, split-conformal) | 80.0 % | nominal 80 %, by construction |
| Prediction interval coverage (test, split-conformal) | 73.1 % | gap is 2020 distribution shift |
| Prediction interval coverage (test, ACI baseline) | 80.1 % | adaptive recalibration closes the gap |
| Test set | 6,576 hours | 2020 Jan–Sep (covers the COVID-19 lockdown) |

## What's in here

```
mlops-forecast/
├── src/mlops_forecast/
│   ├── pipelines/                  # eight Kedro pipelines
│   │   ├── data_quality/           # Great Expectations v1
│   │   ├── data_cleaning/          # timestamp normalisation, IQR outliers
│   │   ├── data_feat_engineering/  # lags, rolling, calendar, Fourier
│   │   ├── data_split/             # strict chronological 70/15/15
│   │   ├── model_train/            # LightGBM + Prophet + Naive, walk-forward CV
│   │   ├── model_selection/        # auto-promote winner to MLflow Production
│   │   ├── model_predict/          # load by registry stage, never by path
│   │   └── data_drifts/            # Evidently, run against two current windows
│   └── serving/quantile_lgbm.py    # pyfunc wrapper: point + conformalised intervals
│
├── api/                            # FastAPI service (Kedro is not in the runtime image)
├── streamlit_app/                  # 4-tab dashboard, consumes the API
├── feature_store/                  # Feast: offline parquet + SQLite online store
│
├── conf/                           # Kedro catalog and parameters
├── tests/                          # pytest, 46 tests including end-to-end pipeline integration
├── docker/                         # multi-stage Dockerfiles
├── docker-compose.yml              # MLflow + API + Streamlit
├── .github/workflows/ci.yml        # lint + test + kedro registry smoke check
├── .pre-commit-config.yaml         # ruff, ruff-format, secret scan, big-file gate
├── scripts/                        # download_data, publish_to_feast, feast_demo
├── report/
│   ├── MLops_Report.tex            # LaTeX source
│   ├── MLops_Report.pdf            # built output
│   └── figures/                    # auto-generated PNGs
└── README.md                       # this file
```

## Run it locally

### macOS / Linux

Five commands take you from a fresh clone to a running stack:

```bash
make install                       # creates .venv, installs the editable package
python scripts/download_data.py    # ~124 MB OPSD CSV → data/01_raw/
make run                           # all eight pipelines (~3 min)
make feast-publish                 # publish features into the Feast online store
make serve                         # docker compose: MLflow + API + Streamlit
```

### Windows

`make` is not available on Windows by default. Use the PowerShell wrapper instead. Unlike the Make targets, the wrapper runs the whole sequence in one pass (create venv → install → download data → all eight pipelines → Feast publish + smoke test → pytest → ruff/mypy/kedro checks); you tune it with switches rather than running individual steps:

```powershell
# Full run (everything except the Docker serving stack):
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1

# Full run, then bring up the Docker serving stack at the end:
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1 -Serve
```

Available switches:

| Switch | Effect |
|---|---|
| `-Serve` | after everything, start the Docker serving stack (MLflow + API + Streamlit). Requires Docker Desktop running |
| `-OnlyData` | download the OPSD dataset only, then stop |
| `-SkipFeast` | skip the Feast publish + smoke test |
| `-SkipTests` | skip the pytest suite |
| `-SkipChecks` | skip the ruff / mypy / `kedro registry list` quality checks |
| `-Recreate` | delete and rebuild the `.venv` from scratch |

Prerequisites on Windows: Python 3.11 (from python.org), Docker Desktop (only for `-Serve`).

| Service | URL | What it shows |
|---|---|---|
| MLflow UI | http://localhost:5000 | runs, metrics, registered model `ElectricityForecast` |
| FastAPI Swagger | http://localhost:8000/docs | live `/predict`, `/health`, `/model/info` |
| Streamlit dashboard | http://localhost:8501 | forecast, explainability, drift, model registry |

A live `/predict` call:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"start_timestamp":"2020-12-01T00:00:00Z","horizon_hours":24}'
```

The response includes `load_mw_predicted` plus the conformalised `lower_bound` and `upper_bound` for each hour.

### Without Docker

If Docker is unavailable the three services run from the venv:

```bash
# MLflow on 5001 (5000 collides with macOS ControlCenter)
.venv/bin/mlflow ui --backend-store-uri file://$PWD/mlruns --port 5001 &

# API
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
  .venv/bin/uvicorn api.main:app --port 8000 &

# Streamlit
API_BASE_URL=http://127.0.0.1:8000 \
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
  .venv/bin/streamlit run streamlit_app/app.py
```

## Architecture

```
                        data/01_raw/opsd_germany_hourly.csv (124 MB)
                                          │
                                          ▼
              ┌────────────────────────────────────────────────┐
              │  data_quality   (Great Expectations v1)        │ → 08_reporting/data_quality/
              │  data_cleaning  (timestamp, NaN, IQR)          │ → 02_intermediate/
              │  data_feat_engineering (lags, rolling,         │ → 04_feature/
              │                          Fourier, calendar)    │
              │  data_split  (chronological 70/15/15)          │ → 05_model_input/
              │  model_train (Naive, Prophet, LightGBM         │ ─MLflow runs─→ ./mlruns/
              │              + walk-forward CV + quantile      │ ─SHAP─→ 08_reporting/explainability/
              │               heads + conformal calibration)   │
              │  model_selection (auto-promote → Production)   │ ─Registry─→ ElectricityForecast/Production
              │  model_predict  (load by registry stage)       │ → 07_model_output/predictions.parquet
              │  data_drifts  (Evidently, two windows)         │ → 08_reporting/drift/
              └────────────────────────────────────────────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                       FastAPI         Feast          Streamlit
                      (port 8000)  (offline+online)  (port 8501)
                          │
                       MLflow registry ← models:/ElectricityForecast/Production
                       (port 5000)
```

## Tech stack

Python 3.11. CI also runs the test suite on Python 3.12.

| Component | Tool | Version |
|---|---|---|
| Orchestration | Kedro + kedro-viz | 0.19.8 / 9+ |
| Experiment tracking | MLflow + kedro-mlflow | 2.15 / 0.12 |
| Feature store | Feast | 0.40+ |
| Data quality | Great Expectations v1 | 1.x |
| Drift detection | Evidently | 0.4+ |
| Models | LightGBM, Prophet | 4.5+ / 1.1.6+ |
| Explainability | SHAP TreeExplainer | 0.46+ |
| Serving | FastAPI, Uvicorn | 0.110+ / 0.30+ |
| Dashboard | Streamlit | 1.30+ |
| CI | GitHub Actions, pytest | — |
| Containers | Docker Compose v2 | — |

### Deviations from the handout

The handout names a reference stack. Three components depart from it on engineering grounds rather than availability constraints; the rest is verbatim.

| Original | Used here | Reason |
|---|---|---|
| NannyML | Evidently | listed as a permitted alternative in the handout, more actively maintained |
| Hopsworks | Feast | removes the API-key dependency so the project reproduces from a clean clone with no external services |
| GE 0.18 | GE 1.x | 1.x is the current production version of Great Expectations; 0.18 is in deprecation |

kedro-mlflow and kedro-viz are installed per the handout. kedro-mlflow's auto-tracking hook is disabled in `conf/base/mlflow.yml` because the training nodes manage their own MLflow runs; the package's dataset wrappers and CLI utilities remain available.

## Engineering principles

A small set of rules, each answering a specific failure mode that bites time-series projects:

1. **No random shuffling.** Train, validation, and test are chronological slices. Shuffling leaks the future into training and silently inflates validation metrics.
2. **No feature leakage.** Rolling statistics use `shift(1)` before the window, not after. Lag features are computed inside each split, never on the concatenated frame.
3. **Drift detection has to be falsifiable.** The drift pipeline runs Evidently against two current-data windows: a within-2019 control and the COVID lockdown. The 3× separation in score magnitude is what carries the signal, not the binary flag.
4. **No hardcoded model paths.** The API loads `models:/ElectricityForecast/Production`. Promotions and rollbacks happen in the registry; the serving image does not need to be rebuilt.
5. **Lean serving image.** The `mlops_forecast.serving` package is what gets pickled into the registry. The API image deliberately excludes Kedro, SHAP, Evidently, and Great Expectations.
6. **Calibrated intervals, not raw quantile heads.** Quantile regression on its own undercovers; we wrap it with split-conformal calibration so that the validation interval coverage matches the nominal 80%.
7. **Determinism.** Every model log records a `train_data_hash` (SHA-256 of the training parquet). Two runs on the same data have the same hash; divergent hashes show up in the MLflow diff view.
8. **Pre-commit gates everything.** ruff, ruff-format, secret detection, large-file detection, merge-conflict detection.

## Make targets

```text
make install          # set up .venv, install the editable package
make download-data    # fetch the 124 MB OPSD CSV
make run              # run all eight pipelines end-to-end
make test             # pytest suite, 46 tests, 82% coverage
make typecheck        # mypy --strict-ish across src/, no warnings allowed
make demo             # one-shot: download data, run pipelines, serve all three apps
make lint             # ruff check + ruff format check
make feast-publish    # publish features into Feast (apply + materialize)
make feast-demo       # smoke-test offline + online retrieval
make serve            # docker compose up (MLflow + API + Streamlit)
make mlflow-ui        # local MLflow UI without Docker
make report-figures   # regenerate the PNGs in report/figures/
make report           # build report/MLops_Report.pdf with tectonic
make clean            # wipe build artefacts (data/ and mlruns/ untouched)
make clean-data       # wipe generated data layers (preserves raw + .gitkeep)
```

## Documentation

- [`report/MLops_Report.pdf`](report/MLops_Report.pdf): the 6-page LaTeX report covering architecture, methodology, results, drift analysis, operations and governance, and risk register.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): branching, commit style, time-series guardrails. Read before contributing.
- [`feature_store/feature_repo/electricity.py`](feature_store/feature_repo/electricity.py): the Feast definitions with rationale in the docstrings.
- Per-pipeline `nodes.py` files: each starts with a docstring that explains the design decision in place.

## License

Educational use, NOVA IMS MLOps 2025/26. The OPSD dataset is CC BY 4.0; see [data.open-power-system-data.org](https://data.open-power-system-data.org/).
