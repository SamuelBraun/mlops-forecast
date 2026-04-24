"""Data quality pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_data_quality_checks


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=run_data_quality_checks,
                inputs=["raw_electricity", "params:data_quality"],
                outputs="data_quality_report",
                name="run_data_quality_checks_node",
                tags=["data_quality"],
            ),
        ]
    )
