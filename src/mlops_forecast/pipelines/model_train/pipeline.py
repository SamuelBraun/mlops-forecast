"""Model training pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_lightgbm, train_prophet, train_seasonal_naive


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=train_seasonal_naive,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:model_train",
                    "split_metadata",
                ],
                outputs="naive_run_info",
                name="train_seasonal_naive_node",
                tags=["model_train", "baseline"],
            ),
            node(
                func=train_lightgbm,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:model_train",
                    "split_metadata",
                ],
                outputs=["lgbm_run_info", "shap_values", "shap_summary_plot"],
                name="train_lightgbm_node",
                tags=["model_train", "lightgbm"],
            ),
            node(
                func=train_prophet,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:model_train",
                    "split_metadata",
                ],
                outputs="prophet_run_info",
                name="train_prophet_node",
                tags=["model_train", "prophet"],
            ),
        ]
    )
