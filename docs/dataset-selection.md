# Dataset Selection

**Selected dataset:** Open Power System Data (OPSD) — Germany Hourly Time Series
**Decision date:** Week of 06/05/2026 (before midterm checkpoint 13/05)

---

## Candidate Evaluation

### 1. OPSD Germany Hourly (SELECTED)
| Attribute | Detail |
|-----------|--------|
| Source | https://data.open-power-system-data.org/time_series/ |
| License | CC BY 4.0 — fully redistributable |
| Granularity | Hourly |
| Length | 2015–2020 (5 years, ~43,800 rows) |
| Type | Multivariate — load + wind + solar generation |
| Exogenous | Wind and solar generation (proxy for weather conditions) |
| Drift potential | **High** — COVID-19 lockdown (Mar–May 2020) caused measurable structural break |

**Why selected:** Electricity load is one of the best-studied and most teachable time series domains. The dataset has strong, interpretable seasonality (intraday, weekly, annual), natural exogenous features, and a known structural break (COVID-19) that makes drift detection demonstrable without artificial injection. The CC BY 4.0 license allows us to document the download URL rather than committing the data.

---

### 2. Rossmann Store Sales (Rejected)
| Attribute | Detail |
|-----------|--------|
| Source | Kaggle (Rossmann challenge) |
| License | Competition terms — restricted redistribution |
| Granularity | Daily |
| Length | ~2.5 years, 1,115 stores |
| Type | Panel (multivariate, hierarchical) |

**Rejected because:** License restrictions complicate sharing. Panel structure adds modelling complexity that distracts from the MLOps focus. Daily granularity gives fewer rows per unit time, making drift monitoring less visually convincing.

---

### 3. Metro Interstate Traffic Volume (UCI)
| Attribute | Detail |
|-----------|--------|
| Source | UCI ML Repository |
| License | CC0 Public Domain |
| Granularity | Hourly |
| Length | 2012–2018 (~48,000 rows) |
| Type | Univariate + weather exogenous |

**Rejected because:** Only univariate target with limited feature richness. Weather exogenous features require a separate data join. Less globally recognisable domain than electricity.

---

### 4. M5 Forecasting Competition (Walmart)
| Attribute | Detail |
|-----------|--------|
| Source | Kaggle |
| License | Competition terms |
| Granularity | Daily |
| Length | 5 years, 42,840 series |

**Rejected because:** Enormous scale (42K series) exceeds Pandas feasibility for a course project. Would require Spark mitigation immediately, consuming sprint time that should go to MLOps tooling.

---

### 5. Air Quality Dataset (Beijing PM2.5)
| Attribute | Detail |
|-----------|--------|
| Source | UCI ML Repository |
| License | CC0 |
| Granularity | Hourly |
| Length | 2010–2014 (~43,000 rows) |

**Rejected because:** Missing value rates >20% require extensive imputation work that dominates sprint time. Target variable (PM2.5 concentration) is harder for a general audience to relate to.

---

## Selected Dataset: Technical Details

**Download URL:**
```
https://data.open-power-system-data.org/time_series/latest/time_series_60min_singleindex.csv
```

**Columns used:**

| Raw column name | Internal name | Description |
|----------------|---------------|-------------|
| `utc_timestamp` | `timestamp` | UTC hourly timestamp |
| `DE_load_actual_entsoe_transparency` | `load_mw` | **Target** — Germany electricity load in MW |
| `DE_wind_generation_actual` | `wind_mw` | Wind power generation in MW |
| `DE_solar_generation_actual` | `solar_mw` | Solar PV generation in MW |

**Approximate statistics (2015–2020):**
- Mean load: ~55,000 MW
- Min load: ~35,000 MW (summer nights)
- Max load: ~80,000 MW (winter weekday mornings)
- Load standard deviation: ~9,500 MW

---

## Success Metrics

### Primary metric: MAPE (Mean Absolute Percentage Error)

$$\text{MAPE} = \frac{1}{n}\sum_{t=1}^{n} \left|\frac{y_t - \hat{y}_t}{y_t}\right| \times 100\%$$

**Why MAPE for this problem:**
- Interpretable in business terms: "our forecast is off by X% on average"
- Standard metric in electricity demand forecasting literature
- Unitless, so comparable across load levels and time periods
- Industry benchmark for grid operators: ≤5% MAPE is "good", ≤3% is "excellent"

**Caveat:** MAPE is undefined when actual = 0 (never in this dataset — load always > 30,000 MW) and penalises under-forecasting more than over-forecasting. We also report sMAPE and RMSE.

### Secondary metrics:
- **sMAPE** — symmetric, handles near-zero values better
- **RMSE (MW)** — absolute error in original units (useful for capacity planning)

### Baseline forecasts (must beat both):
1. **Seasonal Naive (168h):** Use load from the same hour one week ago → estimated MAPE ~8-12%
2. **Daily mean:** Use the average load for that hour across all training data → estimated MAPE ~15-20%

A LightGBM model with lag features should achieve MAPE ≤5% on the validation set.

---

## Drift Demonstration Design

The COVID-19 lockdown provides a **natural experiment** for drift detection:

| Period | Expected behaviour |
|--------|--------------------|
| 2019 (reference) | Normal load patterns — no drift |
| 2020 Q1 (Jan–Mar 15) | Pre-lockdown — no significant drift |
| 2020 Mar 16 – May 15 | German lockdown — significant load reduction (~5-10%), shift in intraday profile |
| 2020 May 16 onward | Partial recovery |

The `data_drifts` pipeline will run the Evidently drift calculator on both windows and compare results, proving the monitor is **falsifiable** — it reports both no-drift and drift-detected cases.
