"""Publish the Kedro feature parquet into the Feast feature store.

Reads ``data/04_feature/features_electricity.parquet`` (Kedro output), reshapes
it into a Feast-compatible layout (timestamp as column, entity column), writes
``feature_store/data/electricity_features.parquet``, then runs ``feast apply``
and materializes into the SQLite online store.

Run after every training run that produces fresh features::

    python scripts/publish_to_feast.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("publish_to_feast")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KEDRO_FEATURES = PROJECT_ROOT / "data" / "04_feature" / "features_electricity.parquet"
FEAST_DATA_DIR = PROJECT_ROOT / "feature_store" / "data"
FEAST_PARQUET = FEAST_DATA_DIR / "electricity_features.parquet"
FEAST_REPO = PROJECT_ROOT / "feature_store" / "feature_repo"

# Use the feast CLI from the same interpreter that's running this script. Works
# inside venvs even when the user hasn't activated them. On Windows the console
# script is `feast.exe`, on POSIX it's `feast` (no extension).
def _find_feast_bin() -> Path | None:
    scripts_dir = Path(sys.executable).parent
    for name in ("feast.exe", "feast"):
        candidate = scripts_dir / name
        if candidate.exists():
            return candidate
    return None


FEAST_BIN = _find_feast_bin()


def main() -> None:
    if not KEDRO_FEATURES.exists():
        raise SystemExit(
            f"Feature parquet not found at {KEDRO_FEATURES}. "
            "Run `kedro run --pipeline data_feat_engineering` first."
        )

    logger.info("Reading Kedro features from %s", KEDRO_FEATURES)
    df = pd.read_parquet(KEDRO_FEATURES)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in df.columns:
        # Index might come back unnamed depending on the writer
        df = df.rename(columns={df.columns[0]: "timestamp"})

    # Feast wants a tz-naive UTC timestamp on disk for FileSource.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df["grid_zone"] = "DE"

    FEAST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEAST_PARQUET, index=False)
    logger.info("Wrote %d rows to %s", len(df), FEAST_PARQUET)

    logger.info("Running `feast apply`")
    subprocess.run([str(FEAST_BIN), "apply"], cwd=FEAST_REPO, check=True)

    # The OPSD dataset ends in 2020. Too old for `materialize-incremental`
    # (it expects fresh data relative to wall clock). Use explicit start/end
    # bounded to the actual data range.
    start = df["timestamp"].min().isoformat()
    end = df["timestamp"].max().isoformat()
    logger.info("Materializing online store from %s to %s", start, end)
    subprocess.run(
        [str(FEAST_BIN), "materialize", start, end],
        cwd=FEAST_REPO,
        check=True,
    )
    logger.info("Feast feature store ready (registry: %s/data/registry.db)", FEAST_REPO.parent)


if __name__ == "__main__":
    if FEAST_BIN is None:
        raise SystemExit(
            f"feast CLI not found in {Path(sys.executable).parent} "
            "(looked for feast.exe / feast). Install with `pip install -e .`"
        )
    main()
