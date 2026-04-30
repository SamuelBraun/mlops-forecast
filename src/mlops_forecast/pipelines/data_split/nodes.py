"""Strict chronological train/val/test split.

The split is by timestamp boundary. We never use `train_test_split` or any
shuffle on this dataset: random splitting on a time series puts future
points in the training set, gives optimistic validation numbers, and then
fails silently when the model meets fresh data in production.
"""

from __future__ import annotations

import hashlib
import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split the feature matrix into train/val/test sets chronologically.

    Cutoffs from parameters.yml:
      - train: start → train_end
      - val:   train_end + 1h → val_end
      - test:  val_end + 1h → end (includes COVID lockdown period for drift demo)

    Args:
        df: Full feature matrix with DatetimeIndex and 'load_mw' target.
        params: `data_split` section from parameters.yml.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata_json).
    """
    target = "load_mw"
    train_end = pd.Timestamp(params["train_end"], tz="UTC")
    val_end = pd.Timestamp(params["val_end"], tz="UTC")

    train_mask = df.index <= train_end
    val_mask = (df.index > train_end) & (df.index <= val_end)
    test_mask = df.index > val_end

    train = df[train_mask]
    val = df[val_mask]
    test = df[test_mask]

    feature_cols = [c for c in df.columns if c != target]

    X_train, y_train = train[feature_cols], train[[target]]
    X_val, y_val = val[feature_cols], val[[target]]
    X_test, y_test = test[feature_cols], test[[target]]

    # Belt-and-braces: assert the boundaries actually hold. If somebody
    # tweaks the cutoffs in parameters.yml and overlaps the windows, this
    # is what catches it.
    assert X_train.index.max() < X_val.index.min(), "Train/val boundary violated!"
    assert X_val.index.max() < X_test.index.min(), "Val/test boundary violated!"

    logger.info(
        "Split: train=%d rows (%s → %s), val=%d rows, test=%d rows",
        len(train),
        train.index.min().date(),
        train.index.max().date(),
        len(val),
        len(test),
    )

    # MD5 of the training rows. Logged to every MLflow run so two runs on
    # the same data have the same hash, and a silent change in inputs is
    # visible in the MLflow diff view.
    train_hash = hashlib.md5(
        pd.util.hash_pandas_object(X_train, index=True).values.tobytes()
    ).hexdigest()

    metadata = {
        "train_start": str(X_train.index.min()),
        "train_end": str(X_train.index.max()),
        "val_start": str(X_val.index.min()),
        "val_end": str(X_val.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "train_data_hash": train_hash,
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, json.dumps(metadata, indent=2)
