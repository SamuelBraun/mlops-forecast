"""Data split pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import temporal_split


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=temporal_split,
                inputs=["features_electricity", "params:data_split"],
                outputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "X_test",
                    "y_test",
                    "split_metadata",
                ],
                name="temporal_split_node",
                tags=["data_split"],
            ),
        ]
    )
