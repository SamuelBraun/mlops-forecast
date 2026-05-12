"""Lean modules consumed at *serving* time, not training time.

Anything pickled into an MLflow registry artifact (model wrappers, pyfunc
classes) lives here so the inference container doesn't need to import Kedro.
"""
