import numpy as np
import pandas as pd
import pytest
import mlflow
from sklearn.linear_model import LinearRegression

from src.train import select_best, train_and_log


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "SEX": rng.choice(["F", "M"], n),
        "PCS_ESE": rng.choice(["1T3", "4", "5", "6"], n),
        "DEP": rng.choice(["75", "69", "13"], n),
        "TIME_PERIOD": rng.choice([2023, 2024], n),
    })
    y = pd.Series(rng.normal(2500, 400, n))
    split = 40
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def test_train_and_log_creates_mlflow_run_with_metrics_and_artifact(tmp_path, toy_data):
    X_train, X_test, y_train, y_test = toy_data
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    mlflow.set_experiment("test-experiment")

    result = train_and_log(
        "regression_lineaire", LinearRegression(), {}, X_train, y_train, X_test, y_test, tmp_path
    )

    assert result["name"] == "regression_lineaire"
    assert result["test_rmse"] >= 0
    assert "pipeline" in result
    assert (tmp_path / "regression_lineaire_pred_vs_actual.png").exists()

    run = mlflow.get_run(result["run_id"])
    assert run.data.params["model_name"] == "regression_lineaire"
    assert "test_rmse" in run.data.metrics
    assert "test_mae" in run.data.metrics
    assert "cv_rmse" in run.data.metrics


def test_select_best_excludes_reference_and_picks_lowest_rmse():
    results = [
        {"name": "reference_moyenne", "test_rmse": 10.0},
        {"name": "regression_lineaire", "test_rmse": 5.0},
        {"name": "random_forest", "test_rmse": 3.0},
    ]
    best = select_best(results)
    assert best["name"] == "random_forest"
