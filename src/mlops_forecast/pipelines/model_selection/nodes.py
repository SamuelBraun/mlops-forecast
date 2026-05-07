"""Pick the winning model and walk it through the registry stages.

Compares the three candidate runs by validation MAPE, finds the matching
MLflow registry version, and walks it None -> Staging -> Production. We
auto-promote in this demo because there is no human reviewer in the loop;
in a deployed system the Staging -> Production transition would sit
behind a manual approval.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

# File-based MLflow store under `./mlruns`. See model_train/nodes.py for
# the same default used in the trainer.
_DEFAULT_MLFLOW_URI = f"file://{Path('mlruns').resolve()}"


def select_and_promote(
    naive_run_info: dict,
    lgbm_run_info: dict,
    prophet_run_info: dict,
    params: dict,
) -> pd.DataFrame:
    """Sort the candidates by `primary_metric` and promote the winner."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        os.makedirs("mlruns", exist_ok=True)
        mlflow.set_tracking_uri(_DEFAULT_MLFLOW_URI)

    primary_metric = params["primary_metric"]
    register_name = "ElectricityForecast"

    candidates = [naive_run_info, lgbm_run_info, prophet_run_info]
    results_df = pd.DataFrame(candidates).sort_values(primary_metric)
    logger.info(
        "Model selection:\n%s", results_df[["model_type", primary_metric, "val_rmse"]].to_string()
    )

    best = results_df.iloc[0]
    best_run_id = best["run_id"]
    logger.info("Best: %s (val_mape=%.2f%%)", best["model_type"], best[primary_metric])

    client = MlflowClient()

    # The trainer registered all three candidates as versions of the same
    # registered model name. Find the version whose run_id matches our
    # winner; everything else gets archived by the transition call below.
    try:
        versions = client.search_model_versions(f"name='{register_name}'")
        best_version = next(
            (
                v
                for v in sorted(versions, key=lambda x: int(x.version), reverse=True)
                if v.run_id == best_run_id
            ),
            None,
        )
        if best_version is None:
            logger.warning("No registered version found for run %s", best_run_id)
            results_df["promoted"] = False
            return results_df

        # Step 1: walk into Staging.
        client.transition_model_version_stage(
            name=register_name,
            version=best_version.version,
            stage="Staging",
            archive_existing_versions=True,
        )
        logger.info("→ Staging: %s v%s", register_name, best_version.version)

        # Step 2: walk into Production. A deployed system would gate this
        # behind a human review of metrics + drift on the Staging model.
        client.transition_model_version_stage(
            name=register_name,
            version=best_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("→ Production: %s v%s", register_name, best_version.version)

    except Exception as exc:  # noqa: BLE001
        logger.error("Model promotion failed: %s", exc)

    results_df["promoted"] = results_df["run_id"] == best_run_id
    return results_df
