"""Entraînement, suivi MLflow et sélection du meilleur modèle."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.data import build_preprocessor, prepare_dataset

DATA_PATH = Path("data/raw/DS_BTS_SAL_EQTP_SEX_PCS_2024_data.csv")
MODELS_DIR = Path("models")
EXPERIMENT_NAME = "salaires-eqtp-prediction"


def get_model_configs() -> dict:
    """Modèles à comparer, avec des hyperparamètres fixes raisonnables (pas de recherche de grille)."""
    return {
        "reference_moyenne": (DummyRegressor(strategy="mean"), {"strategy": "mean"}),
        "regression_lineaire": (LinearRegression(), {}),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=150, max_depth=12, min_samples_leaf=10, random_state=42, n_jobs=-1
            ),
            {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 10},
        ),
        "gradient_boosting": (
            HistGradientBoostingRegressor(
                learning_rate=0.1, max_iter=200, max_leaf_nodes=31, random_state=42
            ),
            {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31},
        ),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("modele", model)])


def predicted_vs_actual_plot(y_test, y_pred, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, y_pred, alpha=0.2, s=8)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "k--", linewidth=1)
    ax.set_xlabel("Salaire réel (€)")
    ax.set_ylabel("Salaire prédit (€)")
    ax.set_title("Prédit vs réel")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def train_and_log(name, model, params, X_train, y_train, X_test, y_test, artifacts_dir: Path) -> dict:
    """Entraîne un pipeline, logue params/métriques/artefacts dans un run MLflow, retourne le résumé."""
    pipeline = build_pipeline(model)
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name)
        for key, value in params.items():
            mlflow.log_param(key, value)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        cv_rmse = float(np.mean(-cross_val_score(
            build_pipeline(model), X_train, y_train, cv=3, scoring="neg_root_mean_squared_error"
        )))

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)
        mlflow.log_metric("cv_rmse", cv_rmse)

        plot_path = Path(artifacts_dir) / f"{name}_pred_vs_actual.png"
        predicted_vs_actual_plot(y_test, y_pred, plot_path)
        mlflow.log_artifact(str(plot_path))

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = mlflow.active_run().info.run_id

    return {
        "name": name,
        "run_id": run_id,
        "pipeline": pipeline,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
        "cv_rmse": cv_rmse,
    }


def select_best(results: list) -> dict:
    """Choisit le run avec la meilleure RMSE test, en excluant la référence."""
    candidats = [r for r in results if r["name"] != "reference_moyenne"]
    return min(candidats, key=lambda r: r["test_rmse"])


def run_training(data_path: Path = DATA_PATH, models_dir: Path = MODELS_DIR) -> dict:
    """Point d'entrée : entraîne tous les modèles, logue dans MLflow, exporte le meilleur."""
    models_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = prepare_dataset(data_path)

    results = []
    for name, (model, params) in get_model_configs().items():
        print(f"Entraînement : {name}...")
        results.append(train_and_log(name, model, params, X_train, y_train, X_test, y_test, models_dir))

    best = select_best(results)
    joblib.dump(best["pipeline"], models_dir / "best_model.joblib")

    model_card = {
        "model_name": best["name"],
        "run_id": best["run_id"],
        "test_mae": best["test_mae"],
        "test_rmse": best["test_rmse"],
        "test_r2": best["test_r2"],
        "categories_vues": {
            "SEX": sorted(X_train["SEX"].unique().tolist()),
            "PCS_ESE": sorted(X_train["PCS_ESE"].unique().tolist()),
            "DEP": sorted(X_train["DEP"].unique().tolist()),
        },
        "annees_vues": sorted(int(a) for a in X_train["TIME_PERIOD"].unique().tolist()),
        "tous_les_runs": [
            {"name": r["name"], "test_mae": r["test_mae"], "test_rmse": r["test_rmse"], "test_r2": r["test_r2"]}
            for r in results
        ],
    }
    with open(models_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, ensure_ascii=False, indent=2)

    return model_card


if __name__ == "__main__":
    card = run_training()
    print(f"\nMeilleur modèle : {card['model_name']} (RMSE test = {card['test_rmse']:.1f} €, R² = {card['test_r2']:.2f})")
