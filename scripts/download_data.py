"""Download the OPSD Germany hourly time series CSV into data/01_raw/."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://data.open-power-system-data.org/time_series/latest/"
    "time_series_60min_singleindex.csv"
)
DEST = Path("data/01_raw/opsd_germany_hourly.csv")


def download() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Already exists: {DEST}. Delete it to re-download.")
        sys.exit(0)

    print(f"Downloading OPSD dataset from {URL} ...")
    urlretrieve(URL, DEST)  # noqa: S310
    size_mb = DEST.stat().st_size / 1_048_576
    print(f"Downloaded {size_mb:.1f} MB → {DEST}")


if __name__ == "__main__":
    download()
