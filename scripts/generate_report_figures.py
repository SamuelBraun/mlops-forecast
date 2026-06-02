"""Generate the figures that ship in the LaTeX report.

Produces four PNGs in ``report/figures/``:

* ``drift_comparison.png``: clean vs COVID drift scores side by side
* ``monthly_mape.png``: monthly MAPE with the concept-drift threshold band
* ``cv_folds.png``: walk-forward CV MAPE per fold
* ``model_comparison.png``: validation MAPE per candidate model

Run after ``make run`` so the input CSVs and parquets exist.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)

CLEAN = "#1f77b4"
PERTURB = "#d62728"


def fig_drift_comparison() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data/08_reporting/drift/drift_results.csv")
    pivot = df.pivot(index="feature", columns="window", values="drift_score").sort_values(
        "covid_lockdown",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y = range(len(pivot))
    ax.barh(
        [i - 0.18 for i in y],
        pivot["clean_2019_h2"],
        height=0.36,
        color=CLEAN,
        label="2019 H2 (control)",
    )
    ax.barh(
        [i + 0.18 for i in y],
        pivot["covid_lockdown"],
        height=0.36,
        color=PERTURB,
        label="2020 lockdown (perturbed)",
    )
    ax.axvline(0.1, color="grey", linestyle="--", linewidth=1, label="Evidently threshold (0.10)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Wasserstein distance (normalised)")
    ax.set_title(
        "Per-feature drift score: control window vs lockdown window\n"
        "Both reference 2019 H1. Lockdown scores are about 3x larger.",
        fontsize=10,
        loc="left",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    plt.savefig(FIG_DIR / "drift_comparison.png")
    plt.close(fig)


def fig_monthly_mape() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data/08_reporting/drift/concept_drift_results.csv")
    df["mape"] = df["mape"].astype(float)
    df["baseline_mape"] = df["baseline_mape"].astype(float)
    threshold = df["baseline_mape"].iloc[0] * 1.5

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [PERTURB if f else CLEAN for f in df["drift_flag"]]
    ax.bar(df["month"], df["mape"], color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(
        df["baseline_mape"].iloc[0],
        color="grey",
        linestyle="-",
        linewidth=1,
        label=f"Baseline median ({df['baseline_mape'].iloc[0]:.2f}%)",
    )
    ax.axhline(
        threshold,
        color="grey",
        linestyle="--",
        linewidth=1,
        label=f"1.5× threshold ({threshold:.2f}%)",
    )
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title(
        "Monthly test MAPE: April 2020 (lockdown peak) breaches the threshold",
        fontsize=10,
        loc="left",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    plt.savefig(FIG_DIR / "monthly_mape.png")
    plt.close(fig)


def fig_cv_folds() -> None:
    """Pull the four nested CV runs from MLflow and plot per-fold MAPE."""
    mlflow.set_tracking_uri(f"file://{(PROJECT_ROOT / 'mlruns').resolve()}")
    from mlflow import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name("ElectricityForecast")
    if exp is None:
        return
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    folds: list[tuple[int, float]] = []
    for r in runs:
        name = r.data.tags.get("mlflow.runName", "")
        if name.startswith("fold_"):
            idx = int(name.split("_")[1])
            mape = r.data.metrics.get("fold_mape")
            if mape is not None:
                folds.append((idx, mape))
    if not folds:
        return
    folds.sort()

    mean = sum(m for _, m in folds) / len(folds)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot([f[0] for f in folds], [f[1] for f in folds], marker="o", color=CLEAN, linewidth=2)
    ax.axhline(mean, color="grey", linestyle="--", linewidth=1, label=f"Mean {mean:.2f}%")
    ax.set_xticks([f[0] for f in folds])
    ax.set_xlabel("Walk-forward fold (expanding window)")
    ax.set_ylabel("Validation MAPE (%)")
    ax.set_title("Walk-forward CV: stable performance across time", fontsize=10, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    plt.savefig(FIG_DIR / "cv_folds.png")
    plt.close(fig)


def fig_model_comparison() -> None:
    """Bar chart of the three candidate models on val MAPE."""
    df = pd.read_csv(PROJECT_ROOT / "data/06_models/model_selection_results.csv")
    df = df.sort_values("val_mape", ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    colors = [CLEAN if i < len(df) - 1 else PERTURB for i in range(len(df))]
    bars = ax.barh(df["model_type"], df["val_mape"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df["val_mape"]):
        ax.text(
            val + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9
        )
    ax.set_xlabel("Validation MAPE (%), lower is better")
    ax.set_title("Model comparison on the 2019 validation set", fontsize=10, loc="left")
    plt.savefig(FIG_DIR / "model_comparison.png")
    plt.close(fig)


def main() -> None:
    fig_drift_comparison()
    fig_monthly_mape()
    fig_cv_folds()
    fig_model_comparison()
    print(f"Wrote figures to {FIG_DIR}")
    for f in sorted(FIG_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
