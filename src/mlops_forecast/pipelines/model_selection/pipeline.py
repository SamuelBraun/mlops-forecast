"""Model selection pipeline definition."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_and_promote


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=select_and_promote,
                inputs=[
                    "naive_run_info",
                    "lgbm_run_info",
                    "prophet_run_info",
                    "params:model_selection",
                ],
                outputs="model_selection_results",
                name="select_and_promote_node",
                tags=["model_selection"],
            ),
        ]
    )
