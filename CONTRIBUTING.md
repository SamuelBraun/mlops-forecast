# Contributing to MLOps Forecast

## Branching Strategy

| Prefix   | Use case                        | Example                         |
|----------|---------------------------------|---------------------------------|
| `feat/`  | New pipeline or feature         | `feat/model-train-lightgbm`     |
| `fix/`   | Bug fix                         | `fix/data-split-leakage`        |
| `chore/` | Tooling, deps, CI               | `chore/pin-lightgbm-version`    |
| `docs/`  | Documentation only              | `docs/add-feature-store-schema` |
| `test/`  | Tests only                      | `test/add-drift-pipeline-tests` |

**One PR per pipeline.** Each of the eight Kedro pipelines is its own feature branch.

## Commit Style — Conventional Commits

```
<type>(<scope>): <imperative summary>

[optional body explaining WHY, not what]
```

Types: `feat` `fix` `refactor` `test` `docs` `chore` `perf`

Scope = pipeline name or component: `data_quality`, `model_train`, `api`, `dashboard`, `ci`

Examples:
```
feat(data_feat_engineering): add Fourier terms for dual seasonality
fix(data_split): compute rolling features within window to prevent leakage
test(api): add TestClient smoke test for /predict endpoint
docs(report): complete risk analysis section
chore(deps): upgrade evidently to 0.7.21
```

## Pull Request Rules

1. Link to a GitHub Issue in the PR description
2. Fill in the PR template fully (no empty sections)
3. At least **one teammate review** and approval
4. All CI checks must be green before merge (lint, tests, `kedro registry list`)
5. **Do not merge your own PR** — someone else must click the button

## Code Quality

- **Ruff** for linting and import sorting: `make lint`
- **Mypy** for type-checking: `make typecheck`
- **Pytest** with ≥ 60 % coverage on `src/`: `make test`
- All public functions must have type hints
- No TODO comments committed to `main`

## Time Series Safety Checklist

Before submitting any PR that touches feature engineering or model training:

- [ ] Features at time *t* use only data ≤ *t* (no future leakage)
- [ ] No `train_test_split` with `shuffle=True` anywhere
- [ ] Rolling/expanding features are computed inside the training window
- [ ] Target encoding (if used) uses a time-aware fold scheme
- [ ] Pipeline tests cover the leakage guard explicitly

## Data & Secrets Policy

- **Never commit** `data/01_raw/` contents (only `.gitkeep` and sample fixtures)
- **Never commit** `mlruns/`, `mlartifacts/`, or any `.env` file
- **Never commit** `conf/local/credentials.yml` — copy from `credentials.yml.example`
- If you accidentally commit a secret: rotate the key, then use `git filter-repo` to purge

## Sprint Planning Reference

| Week         | Sprint goal                                          | Key deliverable           |
|--------------|------------------------------------------------------|---------------------------|
| 21–28 Apr    | Repo setup, dataset locked, GE suite green           | `data_quality` pipeline   |
| 29 Apr–5 May | Cleaning + feature engineering + Feast publish       | `data_feat_engineering`   |
| 6–12 May     | All pipelines stubbed, CI green, midterm checkpoint  | Pipeline registry         |
| 13–19 May    | Model training + MLflow registry, SHAP               | `model_train` + artifacts |
| 20–26 May    | FastAPI + Docker, model in Production stage          | `make serve` working      |
| 27 May–2 Jun | Drift pipeline, Streamlit dashboard                  | End-to-end demo           |
| 3–30 Jun     | Polish, reproducibility test, report, presentation   | Final submission          |
