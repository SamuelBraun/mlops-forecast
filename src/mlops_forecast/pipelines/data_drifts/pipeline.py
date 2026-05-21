"""Drift detection pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import compute_concept_drift, compute_univariate_drift


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=compute_univariate_drift,
                inputs=["features_electricity", "predictions", "params:data_drifts"],
                outputs=["drift_results", "drift_report_reference", "drift_report_analysis"],
                name="compute_univariate_drift_node",
                tags=["drift", "evidently"],
            ),
            node(
                func=compute_concept_drift,
                inputs=["predictions", "params:data_drifts"],
                outputs="concept_drift_results",
                name="compute_concept_drift_node",
                tags=["drift"],
            ),
        ]
    )
