"""Suppress the Kedro / Python 3.13 import-time warning before any test runs.

Kedro raises a KedroPythonVersionWarning when its package is first imported
on Python 3.13. This file is loaded before any test module, so the filter
goes in early enough to take effect.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="Kedro is not yet fully compatible",
)
