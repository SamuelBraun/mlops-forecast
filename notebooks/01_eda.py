"""Exploratory data analysis on the OPSD Germany hourly time series.

This script uses Jupytext-style ``# %%`` cell markers, which VS Code's
Jupyter extension and JupyterLab both render as a notebook. We keep the
analysis as a script (rather than a checked-in .ipynb) so diffs stay
readable and outputs don't bloat the repo.

Run from the project root after `make install` and `python scripts/download_data.py`:

    .venv/bin/python notebooks/01_eda.py        # as a regular script
    # or open it in VS Code and click "Run Cell" above each # %% block.
"""

# %% [markdown]
# # OPSD Germany Hourly: dataset exploration
#
# We pick this dataset because it has three properties we need for an MLOps
# project: strong multi-period seasonality (intraday, weekly, yearly), real
# exogenous variables (wind and solar generation), and a known structural
# break (the 2020 COVID lockdown) that gives us a real drift case to detect.
#
# The cells below establish those three properties from the data, and pull
# out the few facts that drove design decisions in the cleaning and feature
# engineering pipelines.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "01_raw" / "opsd_germany_hourly.csv"

assert RAW_CSV.exists(), (
    f"Raw OPSD CSV not found at {RAW_CSV}. " "Run `python scripts/download_data.py` first."
)

# %% [markdown]
# ## Loading the raw frame
#
# The OPSD CSV ships with 200+ columns, one per European country/series.
# Loading it without filtering produces a wide and very sparse frame: most
# countries have stretches of full NaN in the early years before SCADA
# coverage arrived, so a naive `dropna()` returns zero rows. The cleaning
# node fixes this by selecting just the three German columns we model.

# %%
df_raw = pd.read_csv(RAW_CSV)
print(f"Raw shape: {df_raw.shape}")
print(f"Columns containing DE_ : {sum(c.startswith('DE_') for c in df_raw.columns)}")
print(f"Total columns: {len(df_raw.columns)}")

# Demonstrate the dropna trap.
print(f"\nNaive dropna on all columns leaves: {len(df_raw.dropna())} rows")
print(
    f"After selecting only the three DE columns we use: "
    f"{len(df_raw[['utc_timestamp', 'DE_load_actual_entsoe_transparency', 'DE_wind_generation_actual', 'DE_solar_generation_actual']].dropna())} rows"
)

# %% [markdown]
# ## Project schema
#
# We rename to the internal names early so the rest of the analysis reads
# cleanly. Pipeline does the same in `data_cleaning.normalise_columns`.

# %%
de = df_raw[
    [
        "utc_timestamp",
        "DE_load_actual_entsoe_transparency",
        "DE_wind_generation_actual",
        "DE_solar_generation_actual",
    ]
].rename(
    columns={
        "utc_timestamp": "timestamp",
        "DE_load_actual_entsoe_transparency": "load_mw",
        "DE_wind_generation_actual": "wind_mw",
        "DE_solar_generation_actual": "solar_mw",
    }
)
de["timestamp"] = pd.to_datetime(de["timestamp"], utc=True)
de = de.set_index("timestamp").sort_index().dropna(subset=["load_mw"])
print(f"After cleaning: {len(de):,} hourly rows from {de.index.min()} to {de.index.max()}")
print(de.describe().round(0))

# %% [markdown]
# ## Three seasonalities, one chart each
#
# Daily, weekly, and yearly cycles are the dominant signal in load data.
# The Fourier features in the engineering pipeline (24h and 168h periods,
# three harmonics each) target exactly the first two of these.

# %%
fig, axes = plt.subplots(3, 1, figsize=(11, 8))

# Daily: average load by hour of day
hourly = de.groupby(de.index.hour)["load_mw"].mean()
axes[0].plot(hourly.index, hourly.values, marker="o")
axes[0].set_title("Average load by hour of day (24h cycle)")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Load (MW)")

# Weekly: average load by day of week
dow = de.groupby(de.index.dayofweek)["load_mw"].mean()
axes[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values)
axes[1].set_title("Average load by day of week (168h cycle)")
axes[1].set_ylabel("Load (MW)")

# Yearly: monthly average per year
yearly = de.groupby([de.index.year, de.index.month])["load_mw"].mean().unstack(0)
yearly.plot(ax=axes[2], colormap="viridis", legend=True)
axes[2].set_title("Monthly average load by year (yearly cycle, year-over-year drift)")
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Load (MW)")
axes[2].legend(title="Year", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## The COVID lockdown shows up clearly at the monthly level
#
# This is the structural break the drift pipeline is designed to detect.
# In the chart below the 2020 line drops well below the 2018 / 2019 lines
# during March / April / May; that is the 80%-confidence drift signal we
# pick up on in the `data_drifts` pipeline.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for year in [2018, 2019, 2020]:
    sub = de[de.index.year == year]
    monthly = sub.groupby(sub.index.month)["load_mw"].mean()
    ax.plot(monthly.index, monthly.values, marker="o", label=str(year))
ax.set_title("Monthly average load: 2020 lockdown is the visible structural break")
ax.set_xlabel("Month")
ax.set_ylabel("Load (MW)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Wind and solar look like sensible exogenous features
#
# Solar is strictly daytime (zero at night, peaks midday in summer); wind
# is non-seasonal at the hour level but has wide variance. Both should
# carry signal that the load regressor can use.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
de.groupby(de.index.hour)["solar_mw"].mean().plot(ax=axes[0], marker="o")
axes[0].set_title("Average solar generation by hour")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("MW")
de.groupby(de.index.month)["wind_mw"].mean().plot(ax=axes[1], marker="o", color="C2")
axes[1].set_title("Average wind generation by month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("MW")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Outlier check (drives `iqr_multiplier` in parameters.yml)
#
# IQR fencing at 3x catches a handful of obvious data-quality glitches
# without trimming legitimate seasonal extremes. Anything outside the box
# below feeds the `remove_outliers` node.

# %%
q1, q3 = de["load_mw"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
mask = (de["load_mw"] < lo) | (de["load_mw"] > hi)
print(f"Q1={q1:.0f}, Q3={q3:.0f}, IQR={iqr:.0f}")
print(f"3x-IQR fence: [{lo:.0f}, {hi:.0f}] MW")
print(f"Rows outside fence: {mask.sum()} ({mask.mean()*100:.3f}% of total)")

# %% [markdown]
# ## Lag autocorrelation justifies the lag set in feature engineering
#
# Yesterday-same-hour and last-week-same-hour are by far the strongest
# linear predictors. The pipeline therefore creates lag features at 1, 2,
# 3, 6, 12, 24, 48, and 168 hours; the SHAP plot in the report later
# confirms `lag_24h` and `lag_168h` dominate at predict time.

# %%
load = de["load_mw"].dropna()
lags = [1, 2, 3, 6, 12, 24, 48, 168, 336, 720]
acf = [load.autocorr(lag=h) for h in lags]
print("Lag autocorrelation:")
for h, c in zip(lags, acf):
    bar = "#" * int(round(abs(c) * 40))
    sign = "+" if c >= 0 else "-"
    print(f"  {h:>4}h: {c:+.3f}  {sign}{bar}")

# %% [markdown]
# ## Takeaways for the rest of the pipeline
#
# 1. **Drop the non-DE columns first.** The full OPSD CSV's NaN pattern
#    makes any later `dropna()` destructive. Done in
#    `data_cleaning.normalise_columns`.
# 2. **Three seasonalities matter.** 24h and 168h get explicit Fourier
#    features; the yearly cycle is captured indirectly by month / week-of-year.
# 3. **The 2020 lockdown is large enough to detect.** It is the perturbed
#    window in `data_drifts.compute_univariate_drift`.
# 4. **Lag features at 24h and 168h are the dominant signal.** The model
#    will lean on them heavily; SHAP later confirms.
# 5. **Outliers are rare and obvious.** A 3x IQR fence is sufficient and
#    cheaper than Isolation Forest.
