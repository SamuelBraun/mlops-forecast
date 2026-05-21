"""Drift detection: univariate feature drift and concept drift via Evidently.

The pipeline runs the same Evidently report against two current-data windows
so the monitor is provably falsifiable: a within-2019 control window where
we don't expect drift, and the 2020 lockdown window where we do.

Originally we planned to use NannyML; it has no Python 3.13 wheel as of
May 2026, and the project handout already lists Evidently as a permitted
alternative. We use Evidently's `legacy` import path because the new
`evidently.metrics` API removed the `DataDriftPreset` we rely on.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

logger = logging.getLogger(__name__)


def _slice_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC"))
    return df[mask].copy()


def _parse_drift_results(report_dict: dict) -> pd.DataFrame:
    """Parse an Evidently report dict into a flat per-column drift DataFrame."""
    rows = []
    for metric in report_dict.get("metrics", []):
        name = metric.get("metric")
        res = metric.get("result", {})
        if name == "DataDriftTable":
            for col_name, col_res in res.get("drift_by_columns", {}).items():
                rows.append(
                    {
                        "feature": col_name,
                        "drift_detected": col_res.get("drift_detected", False),
                        "stattest": col_res.get("stattest_name", ""),
                        "drift_score": col_res.get("drift_score", None),
                        "threshold": col_res.get("stattest_threshold", None),
                    }
                )
        elif name == "ColumnDriftMetric":
            rows.append(
                {
                    "feature": res.get("column_name", ""),
                    "drift_detected": res.get("drift_detected", False),
                    "stattest": res.get("stattest_name", ""),
                    "drift_score": res.get("drift_score", None),
                    "threshold": res.get("stattest_threshold", None),
                }
            )
    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=["feature", "drift_detected", "stattest", "drift_score", "threshold"]
        )
    )


def compute_univariate_drift(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, str, str]:
    """Run Evidently against two windows so the monitor is falsifiable.

    Both windows use the same 2019 H1 reference baseline:

    * `clean_2019_h2` (2019-07 to 2019-12). Same distribution as the
      reference, so we expect minimal drift.
    * `covid_lockdown` (2020-03-15 to 2020-05-15). A real structural break
      in German load due to lockdown measures, so we expect most features
      to drift.

    Both runs are written into `drift_results.csv` tagged by `window`,
    which lets the Streamlit dashboard render them side-by-side. A monitor
    that only ever fires (or never fires) tells you nothing about whether
    it works.
    """
    ref_start = params["reference_start"]
    ref_end = params["reference_end"]
    perturb_start = params.get("perturb_start") or params["analysis_start"]
    perturb_end = params.get("perturb_end") or params["analysis_end"]
    clean_start = params.get("clean_start", "2019-07-01")
    clean_end = params.get("clean_end", "2019-12-31")

    numeric_cols = [c for c in features.select_dtypes(include="number").columns if c != "load_mw"]
    monitor_cols = [
        c
        for c in numeric_cols
        if any(kw in c for kw in ["lag_", "rolling_", "hour", "day_of_week", "is_working"])
    ][:15]
    if not monitor_cols:
        monitor_cols = numeric_cols[:10]

    reference = _slice_period(features, ref_start, ref_end)[monitor_cols].reset_index(drop=True)
    if reference.empty:
        logger.warning("Drift: reference window is empty. Check parameters.yml")
        return pd.DataFrame(), "{}", "{}"

    out_dir = Path("data/08_reporting/drift")
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = [
        (
            "clean_2019_h2",
            clean_start,
            clean_end,
            "Within-2019 control window. We don't expect drift here.",
        ),
        (
            "covid_lockdown",
            perturb_start,
            perturb_end,
            "COVID-19 lockdown. Real structural break, drift expected.",
        ),
    ]

    all_results: list[pd.DataFrame] = []
    summaries: dict[str, dict] = {}
    perturb_dict: dict | None = None

    for label, start, end, note in windows:
        current = _slice_period(features, start, end)[monitor_cols].reset_index(drop=True)
        if current.empty:
            logger.warning("Drift: %s window is empty. Skipping", label)
            continue
        logger.info(
            "Drift [%s]: reference=%d rows, current=%d rows", label, len(reference), len(current)
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        df = _parse_drift_results(report.as_dict())
        df["window"] = label
        all_results.append(df)

        drifted = df.loc[df["drift_detected"] == True, "feature"].tolist()  # noqa: E712
        summaries[label] = {
            "period": f"{start} → {end}",
            "n_rows": len(current),
            "n_features": len(monitor_cols),
            "n_drifted_features": len(drifted),
            "drifted_features": drifted,
            "drift_detected": len(drifted) > 0,
            "note": note,
        }
        if label == "covid_lockdown":
            perturb_dict = summaries[label]

        # Per-window HTML
        report.save_html(str(out_dir / f"evidently_drift_{label}.html"))

        if drifted:
            log = logger.warning if label == "clean_2019_h2" else logger.info
            log(
                "Drift [%s]: %d/%d features drifted: %s",
                label,
                len(drifted),
                len(monitor_cols),
                drifted,
            )
        else:
            log = logger.info if label == "clean_2019_h2" else logger.warning
            log("Drift [%s]: no features drifted (%d checked)", label, len(monitor_cols))

    drift_df = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame(
            columns=["feature", "drift_detected", "stattest", "drift_score", "threshold", "window"]
        )
    )

    ref_report = json.dumps(
        {
            "period": f"{ref_start} → {ref_end}",
            "n_rows": len(reference),
            "status": "reference_baseline",
            "n_features": len(monitor_cols),
            "windows_tested": list(summaries.keys()),
        },
        indent=2,
    )

    # The "analysis" report is the perturbed window (back-compat with old catalog name)
    ana_report = json.dumps(perturb_dict or {}, indent=2)

    # Keep the perturbed-window HTML at the historical filename for the
    # Streamlit dashboard's existing iframe embed.
    perturb_html = out_dir / "evidently_drift_covid_lockdown.html"
    if perturb_html.exists():
        (out_dir / "evidently_drift_report.html").write_bytes(perturb_html.read_bytes())

    return drift_df, ref_report, ana_report


def compute_concept_drift(
    predictions: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Detect concept drift by comparing prediction error distributions over time.

    Monthly MAPE is compared to the reference baseline. Months where MAPE
    exceeds 1.5× the reference average are flagged as drifted.

    Args:
        predictions: Output from model_predict with actual vs predicted values.
        params: `data_drifts` section from parameters.yml.

    Returns:
        DataFrame with monthly MAPE and drift flag.
    """
    df = predictions.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df["month"] = df.index.tz_convert(None).to_period("M")
    monthly = (
        df.groupby("month")
        .apply(
            lambda g: (g["abs_error"] / g["load_mw_actual"].abs()).mean() * 100,
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "mape"})
    )
    monthly["month"] = monthly["month"].astype(str)

    # Test predictions only cover 2020, so there are no pre-analysis months
    # to use as a baseline. We use the median MAPE of months *outside* the
    # perturb window instead. That keeps the flag falsifiable: lockdown
    # months should breach the 1.5x threshold, calmer months should not.
    perturb_start = params.get("perturb_start", "")[:7]
    perturb_end = params.get("perturb_end", "")[:7]
    if perturb_start and perturb_end:
        baseline_mask = ~monthly["month"].between(perturb_start, perturb_end)
    else:
        baseline_mask = pd.Series(True, index=monthly.index)
    ref_mape = monthly.loc[baseline_mask, "mape"].median() if baseline_mask.any() else None

    if ref_mape is not None and not pd.isna(ref_mape):
        monthly["baseline_mape"] = ref_mape
        monthly["drift_flag"] = monthly["mape"] > (ref_mape * 1.5)
        drifted = monthly.loc[monthly["drift_flag"], "month"].tolist()
        if drifted:
            logger.warning(
                "Concept drift detected in months: %s (baseline MAPE=%.2f%%)",
                drifted,
                ref_mape,
            )
        else:
            logger.info("No concept drift detected (baseline MAPE=%.2f%%)", ref_mape)
    else:
        monthly["drift_flag"] = False

    return monthly
