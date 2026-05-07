"""Model predict pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import generate_predictions


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=generate_predictions,
                inputs=["X_test", "y_test", "params:model_predict"],
                outputs="predictions",
                name="generate_predictions_node",
                tags=["model_predict"],
            ),
        ]
    )
