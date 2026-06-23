# Presentation Outline — 10–15 Minutes

**Event:** NOVA IMS MLOps Course, Sprint Review / Final Presentation
**Format:** 10–15 min presentation + Q&A
**Audience:** Prof. Nuno Rosa + classmates
**Tools:** Slides (PowerPoint/Google Slides) + live demo

---

## Slide-by-Slide Outline

### Slide 1 — Title (30 sec)
**"MLOps Forecast: A Reproducible Pipeline for Hourly Electricity Load"**
Group X | NOVA IMS MLOps Spring 2026
Speaker notes: introduce the team and flag that there will be a live demo at the end.

---

### Slide 2 — Problem & Why It Matters (1 min)
- Grid operators must schedule generation **24 hours ahead**
- 1% accuracy improvement = tens of millions € in savings annually
- Germany is the largest electricity market in Europe — public data, real stakes
- "Our goal is not just a model. It is a system that keeps the model working in production."

Speaker notes: hook the audience with the business value. Mention that the exact same
code the team ran is the code the grader can clone.

---

### Slide 3 — Architecture Overview (2 min)
Show the pipeline diagram from the README:

```
OPSD CSV → data_quality → data_cleaning → data_feat_engineering → Feast
                                                    ↓
                                              data_split
                                                    ↓
                                           model_train (3 models) → MLflow
                                                    ↓
                                          model_selection → Production
                                                    ↓
                                           model_predict → FastAPI
                                                    ↓
                                           data_drifts → Streamlit
```

Key points:
- Each block is an independent Kedro pipeline — runnable in isolation
- Every intermediate artefact is versioned and persisted
- MLflow registry is the single source of truth for "what's in production"

---

### Slide 4 — Dataset & Features (1 min)
- OPSD Germany hourly, 2015–2020, CC BY 4.0
- Target: `load_mw` (Germany total load in MW)
- Exogenous: `wind_mw`, `solar_mw`
- 40 engineered features: lags (1h–168h), rolling stats, calendar, Fourier terms
- **Critical design decision:** all lag/rolling features shift before rolling — no leakage

Speaker notes: show the `check-leakage` Claude Code command if time permits.

---

### Slide 5 — Experiment Results (1.5 min)
Show MLflow comparison table (screenshot or live):

| Model | val_MAPE | val_RMSE |
|-------|---------|---------|
| Seasonal Naive | ~10.2% | ~5800 MW |
| Prophet | ~6.1% | ~3400 MW |
| **LightGBM** | **~3.8%** | **~2100 MW** |

LightGBM wins. Top SHAP features: `lag_168h`, `lag_24h`, `hour`.

"We beat the seasonal naive baseline by a factor of ~2.7×."

---

### Slide 6 — MLOps Pipeline in Action (2 min — LIVE DEMO)

**Demo sequence:**
1. `kedro run --pipeline data_quality` → show GE pass/fail output
2. Open MLflow UI → show 3 model runs, LightGBM highlighted as Production
3. `curl -X POST http://localhost:8000/predict -d '{"start_timestamp":"2020-06-01T00:00:00Z","horizon_hours":24}'`
4. Open Streamlit → Forecast view, then Drift view

Fallback (if demo fails): show screenshots pre-captured in `docs/screenshots/`.

---

### Slide 7 — Drift Detection (1.5 min)
"We deliberately chose a dataset with a known structural break."

Show the two-window comparison:
- 2019 reference: no drift detected ✓
- 2020 Q1–Q2 (COVID lockdown): drift detected on `load_mw`, `lag_24h` ✓

"A drift monitor that always says 'no drift' is unfalsifiable. We prove ours works."

Show the Evidently HTML report slide / Streamlit Drift view.

---

### Slide 8 — Risks & Mitigations (1 min)
Three key risks to call out explicitly:

| Risk | When it triggers | Mitigation |
|------|----------|------------|
| Pandas doesn't scale | Data > 500 MB | Migrate to Polars/PySpark — nodes are swap-in |
| Hopsworks free tier rate limit | >10 req/min | Local Parquet cache in `data/04_feature/` |
| Model stale after drift | MAPE > 1.5× reference | Drift alert → manual retrain trigger (automate next sprint) |

---

### Slide 9 — Conclusions & Next Steps (30 sec)
- Reproducible end-to-end MLOps system in 8 weeks
- All components integrated: quality → features → train → serve → monitor → explain
- Next: automate retrain trigger, quantile regression for prediction intervals, K8s

---

### Slide 10 — Q&A Preparation (in presenter notes only)

**Anticipated questions and answers:**

**Q: "Why LightGBM over a deep learning model?"**
A: For this problem and dataset size, LightGBM with lag features consistently
outperforms or matches LSTM in practice, trains 10× faster, and is far more
interpretable via SHAP. The MLOps grade is on system maturity, not model complexity.

**Q: "What fails first if traffic 10×?"**
A: The single-worker FastAPI service. Fix: add Gunicorn workers, then if model inference
becomes the bottleneck, cache predictions for the most-requested horizons or switch to async.

**Q: "How does a new model version reach Production?"**
A: Train → MLflow registers the version → `model_selection` promotes to Staging →
human review in MLflow UI → approve → Production. The API reads Production by name,
so no redeploy needed. We demonstrated this by promoting LightGBM over Prophet.

**Q: "How do you know your forecasts are still good in 6 months?"**
A: The `data_drifts` pipeline runs monthly. When MAPE exceeds 1.5× the reference
baseline, the Streamlit dashboard flags it red. Currently requires manual retrain;
the automated trigger would be the next sprint.

**Q: "Why not just use a cloud ML platform like SageMaker?"**
A: For this course the goal is to understand what each component does. Using managed
services is appropriate in production but would obscure the data quality → feature store →
experiment tracking → serving → monitoring chain we're demonstrating.

**Q: "What's the carbon footprint of training?"**
A: LightGBM on 4 years of hourly data trains in ~4 minutes on a laptop. Total compute
is negligible. A production system would schedule retraining overnight to use off-peak
grid power.

---

## Timing Guide

| Section | Slide | Target time |
|---------|-------|-------------|
| Title + problem | 1–2 | 1.5 min |
| Architecture | 3 | 2 min |
| Dataset + features | 4 | 1 min |
| Results | 5 | 1.5 min |
| Live demo | 6 | 2 min |
| Drift | 7 | 1.5 min |
| Risks | 8 | 1 min |
| Conclusions | 9 | 0.5 min |
| **Total** | | **11 min** |

Buffer: 4 minutes for Q&A or demo overrun.
