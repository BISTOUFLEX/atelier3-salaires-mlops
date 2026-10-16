import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.api.main as main_module


def _toy_pipeline():
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["SEX", "PCS_ESE", "DEP"]),
        ("num", StandardScaler(), ["TIME_PERIOD"]),
    ])
    pipeline = Pipeline([("prep", prep), ("modele", LinearRegression())])
    X = pd.DataFrame({
        "SEX": ["F", "M", "F", "M"],
        "PCS_ESE": ["1T3", "4", "5", "6"],
        "DEP": ["75", "69", "75", "69"],
        "TIME_PERIOD": [2023, 2023, 2024, 2024],
    })
    y = pd.Series([4000.0, 2600.0, 1900.0, 1800.0])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def client(tmp_path, monkeypatch):
    pipeline = _toy_pipeline()
    joblib.dump(pipeline, tmp_path / "best_model.joblib")
    model_card = {"model_name": "regression_lineaire", "test_rmse": 123.4, "test_mae": 98.7}
    with open(tmp_path / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f)

    monkeypatch.setattr(main_module, "MODELS_DIR", tmp_path)

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_returns_model_metadata(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "regression_lineaire"
    assert body["test_rmse"] == 123.4


def test_predict_returns_salary_and_model_name(client):
    response = client.post("/predict", json={"sex": "F", "pcs": "1T3", "department": "75", "year": 2024})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_salary"], float)
    assert body["model_used"] == "regression_lineaire"


def test_predict_rejects_invalid_sex(client):
    response = client.post("/predict", json={"sex": "X", "pcs": "1T3", "department": "75", "year": 2024})
    assert response.status_code == 422


def test_predict_rejects_invalid_pcs(client):
    response = client.post("/predict", json={"sex": "F", "pcs": "9", "department": "75", "year": 2024})
    assert response.status_code == 422
