"""Feature engineering pipeline definition.

The pipeline writes the feature parquet to the layer-04 catalog entry. Feast
publication is handled separately by ``scripts/publish_to_feast.py`` (or by
``make feast-publish``) so that running Kedro does not require a working
Feast install at every run.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_feature_matrix


def create_pipeline(**kwargs) -> Pipeline:  # type: ignore[no-untyped-def]
    return pipeline(
        [
            node(
                func=build_feature_matrix,
                inputs=["clean_electricity", "params:feature_engineering"],
                outputs="features_electricity",
                name="build_feature_matrix_node",
                tags=["feature_engineering"],
            ),
        ]
    )
