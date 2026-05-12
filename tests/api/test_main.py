"""API smoke tests using FastAPI TestClient.

These tests mock the MLflow model load so they run without a live MLflow server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model() -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.full(24, 55000.0)
    model.metadata.flavors = {"lightgbm": {}, "python_function": {}}
    return model


@pytest.fixture
def mock_model_info() -> dict:
    return {
        "model_name": "ElectricityForecast",
        "model_version": "1",
        "model_stage": "Production",
        "run_id": "abc123",
        "registered_at": "1700000000000",
        "metrics": {"val_mape": 3.5, "val_rmse": 1800.0},
        "flavors": ["lightgbm", "python_function"],
    }


@pytest.fixture
def client(mock_model: MagicMock, mock_model_info: dict) -> TestClient:
    """TestClient with model pre-loaded via module-level patching."""
    import api.main as main_module  # noqa: PLC0415

    with (
        patch.object(main_module, "_MODEL", mock_model),
        patch.object(main_module, "_MODEL_INFO", mock_model_info),
    ):
        from api.main import app  # noqa: PLC0415

        yield TestClient(app)


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestModelInfo:
    def test_returns_model_metadata(self, client: TestClient) -> None:
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "ElectricityForecast"
        assert "model_version" in data
        assert "metrics" in data


class TestPredict:
    def test_returns_forecast(self, client: TestClient) -> None:
        payload = {
            "start_timestamp": "2020-06-01T00:00:00Z",
            "horizon_hours": 24,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast" in data
        assert len(data["forecast"]) == 24

    def test_forecast_values_are_floats(self, client: TestClient) -> None:
        payload = {"start_timestamp": "2020-06-01T00:00:00Z", "horizon_hours": 6}
        resp = client.post("/predict", json=payload)
        for point in resp.json()["forecast"]:
            assert isinstance(point["load_mw_predicted"], float)

    def test_horizon_validation(self, client: TestClient) -> None:
        payload = {"start_timestamp": "2020-06-01T00:00:00Z", "horizon_hours": 0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422  # pydantic validation error

    def test_horizon_max(self, client: TestClient) -> None:
        payload = {"start_timestamp": "2020-06-01T00:00:00Z", "horizon_hours": 169}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_response_includes_model_version(self, client: TestClient) -> None:
        payload = {"start_timestamp": "2020-01-01T00:00:00Z", "horizon_hours": 1}
        resp = client.post("/predict", json=payload)
        assert "model_version" in resp.json()


class TestHealthNoModel:
    def test_503_when_model_not_loaded(self) -> None:
        import api.main as main_module  # noqa: PLC0415

        with (
            patch.object(main_module, "_MODEL", None),
            patch.object(main_module, "_MODEL_INFO", {}),
        ):
            from api.main import app  # noqa: PLC0415

            c = TestClient(app, raise_server_exceptions=False)
            resp = c.get("/health")
            assert resp.status_code == 503
