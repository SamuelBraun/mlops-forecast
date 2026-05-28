"""Smoke test for the Feast feature store.

Demonstrates the two retrieval paths a real ML system needs:

  1. **Offline (training)**: point-in-time correct join between an entity
     dataframe and the offline parquet source. No future leakage.
  2. **Online (serving)**: low-latency lookup from the SQLite online store
     of the most-recently materialized feature values.

If both calls return non-empty values for the entity ``DE``, the feature store
contract works.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from feast import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feast_demo")

REPO = Path(__file__).resolve().parents[1] / "feature_store" / "feature_repo"
FEATURES = [
    "electricity_features_v1:lag_1h",
    "electricity_features_v1:lag_24h",
    "electricity_features_v1:rolling_mean_24h",
]


def main() -> None:
    store = FeatureStore(repo_path=str(REPO))

    # 1. Offline (training) retrieval. Historical join at three timestamps
    entity_df = pd.DataFrame(
        {
            "grid_zone": ["DE", "DE", "DE"],
            "event_timestamp": [
                datetime(2020, 1, 15, 12),
                datetime(2020, 4, 1, 12),
                datetime(2020, 6, 1, 12),
            ],
        }
    )
    historical = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURES,
    ).to_df()
    logger.info("Offline (training) features:\n%s", historical.to_string(index=False))
    assert not historical.dropna().empty, "Offline retrieval returned no values"

    # 2. Online (serving) retrieval. Should hit the SQLite online store
    online = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"grid_zone": "DE"}],
    ).to_dict()
    logger.info("Online (serving) features for DE: %s", online)
    assert any(
        v[0] is not None for k, v in online.items() if k != "grid_zone"
    ), "Online retrieval returned all-null values"
    logger.info("Feast smoke test passed")


if __name__ == "__main__":
    main()
