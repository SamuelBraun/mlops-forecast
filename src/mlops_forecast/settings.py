"""Kedro project settings."""

from typing import Any

# The training nodes start and close their own MLflow runs explicitly.
# kedro-mlflow auto-registers a hook via its `kedro.hooks` entry point
# that opens an additional pipeline-level run, which conflicts with our
# per-node runs (UnboundLocalError or "run is already active"). We keep
# kedro-mlflow installed for its dataset wrappers and CLI but disable
# its hook here.
HOOKS: tuple[Any, ...] = ()
DISABLE_HOOKS_FOR_PLUGINS: tuple[str, ...] = ("kedro-mlflow",)
