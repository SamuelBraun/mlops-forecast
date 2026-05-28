"""Feast feature definitions for the electricity forecast project.

Substitutes for Hopsworks (which is not Python-3.13 compatible). The shape
mirrors what we'd ship in production:

  * one entity (`grid_zone`) — single value `DE` for this project, but
    structured so adding more zones later is mechanical;
  * one offline-source view (`electricity_features_v1`) backed by the
    Kedro-produced parquet at ``data/04_feature/features_electricity.parquet``;
  * one feature service (`electricity_v1`) — the contract the API and the
    training job pin to. Bumping schema = new service version = no silent
    consumer breakage.

Run from the repo root::

    cd feature_store/feature_repo
    feast apply                # register definitions in registry.db
    feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64

# Single entity — we forecast one grid zone (Germany) but the modelling is
# generic; multi-zone is a config change, not a code change.
grid_zone = Entity(name="grid_zone", join_keys=["grid_zone"])

# Offline source: the Kedro feature parquet. The `timestamp_field` must be a
# UTC datetime column — Feast uses it as the event-time for point-in-time joins.
electricity_source = FileSource(
    name="electricity_features_source",
    # `scripts/publish_to_feast.py` produces this Feast-compatible parquet:
    # adds a `grid_zone` entity column and promotes the DatetimeIndex to a
    # `timestamp` column (Feast cannot read a parquet index as event-time).
    path="../data/electricity_features.parquet",
    timestamp_field="timestamp",
)

# All numeric features the model consumes. Listed explicitly (not auto-inferred)
# so changes are visible in code review and the Feast registry.
electricity_features_v1 = FeatureView(
    name="electricity_features_v1",
    entities=[grid_zone],
    ttl=timedelta(days=365),
    schema=[
        Field(name="load_mw", dtype=Float64),
        Field(name="wind_mw", dtype=Float64),
        Field(name="solar_mw", dtype=Float64),
        Field(name="lag_1h", dtype=Float64),
        Field(name="lag_2h", dtype=Float64),
        Field(name="lag_3h", dtype=Float64),
        Field(name="lag_6h", dtype=Float64),
        Field(name="lag_12h", dtype=Float64),
        Field(name="lag_24h", dtype=Float64),
        Field(name="lag_48h", dtype=Float64),
        Field(name="lag_168h", dtype=Float64),
        Field(name="rolling_mean_6h", dtype=Float64),
        Field(name="rolling_mean_12h", dtype=Float64),
        Field(name="rolling_mean_24h", dtype=Float64),
        Field(name="rolling_mean_168h", dtype=Float64),
        Field(name="rolling_std_6h", dtype=Float64),
        Field(name="rolling_std_12h", dtype=Float64),
        Field(name="rolling_std_24h", dtype=Float64),
    ],
    online=True,
    source=electricity_source,
)

# Feature service = versioned contract. Training and serving both pin to this
# name + version; schema-incompatible changes go in v2.
electricity_v1 = FeatureService(
    name="electricity_v1",
    features=[electricity_features_v1],
)
