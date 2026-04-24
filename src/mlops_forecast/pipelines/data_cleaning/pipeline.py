"""Data cleaning pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import enforce_non_negative, normalise_columns, reindex_to_hourly, remove_outliers


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=normalise_columns,
                inputs="raw_electricity",
                outputs="normalised_electricity",
                name="normalise_columns_node",
                tags=["data_cleaning"],
            ),
            node(
                func=reindex_to_hourly,
                inputs=["normalised_electricity", "params:data_cleaning"],
                outputs="reindexed_electricity",
                name="reindex_to_hourly_node",
                tags=["data_cleaning"],
            ),
            node(
                func=remove_outliers,
                inputs=["reindexed_electricity", "params:data_cleaning"],
                outputs="outlier_free_electricity",
                name="remove_outliers_node",
                tags=["data_cleaning"],
            ),
            node(
                func=enforce_non_negative,
                inputs="outlier_free_electricity",
                outputs="clean_electricity",
                name="enforce_non_negative_node",
                tags=["data_cleaning"],
            ),
        ]
    )
