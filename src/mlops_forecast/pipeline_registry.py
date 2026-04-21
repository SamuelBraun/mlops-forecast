"""Register all Kedro pipelines so they can be run by name."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from mlops_forecast.pipelines.data_cleaning import create_pipeline as data_cleaning
from mlops_forecast.pipelines.data_drifts import create_pipeline as data_drifts
from mlops_forecast.pipelines.data_feat_engineering import create_pipeline as data_feat_engineering
from mlops_forecast.pipelines.data_quality import create_pipeline as data_quality
from mlops_forecast.pipelines.data_split import create_pipeline as data_split
from mlops_forecast.pipelines.model_predict import create_pipeline as model_predict
from mlops_forecast.pipelines.model_selection import create_pipeline as model_selection
from mlops_forecast.pipelines.model_train import create_pipeline as model_train


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.

    Returns:
        Mapping of pipeline name → Pipeline object.
        The "__default__" key runs the full end-to-end pipeline.
    """
    dq_pipeline = data_quality()
    dc_pipeline = data_cleaning()
    fe_pipeline = data_feat_engineering()
    ds_pipeline = data_split()
    mt_pipeline = model_train()
    ms_pipeline = model_selection()
    mp_pipeline = model_predict()
    dd_pipeline = data_drifts()

    return {
        "data_quality": dq_pipeline,
        "data_cleaning": dc_pipeline,
        "data_feat_engineering": fe_pipeline,
        "data_split": ds_pipeline,
        "model_train": mt_pipeline,
        "model_selection": ms_pipeline,
        "model_predict": mp_pipeline,
        "data_drifts": dd_pipeline,
        # Convenience composite pipelines
        "ingest": dq_pipeline + dc_pipeline,
        "prepare": fe_pipeline + ds_pipeline,
        "train_and_select": mt_pipeline + ms_pipeline,
        "inference": mp_pipeline + dd_pipeline,
        "__default__": (
            dq_pipeline
            + dc_pipeline
            + fe_pipeline
            + ds_pipeline
            + mt_pipeline
            + ms_pipeline
            + mp_pipeline
            + dd_pipeline
        ),
    }
