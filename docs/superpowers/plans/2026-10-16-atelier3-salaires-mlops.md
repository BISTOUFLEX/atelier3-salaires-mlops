# Atelier pratique 3 — Pipeline ML + MLflow + Docker + API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Livrer un repository complet pour l'atelier pratique 3 (Module 3 MLOps) : pipeline ML entraîné sur le dataset salaires Insee (module 2), suivi des essais avec MLflow, API FastAPI de prédiction, image Docker testée localement, et push final vers un repo GitHub privé.

**Architecture:** `src/data.py` (chargement + feature engineering, partagé) → `src/train.py` (4 modèles comparés, logués dans MLflow, meilleur exporté en joblib) → `src/api/` (FastAPI qui sert le modèle exporté) → `Dockerfile` (empaquette code + modèle + dépendances pour la seule inférence).

**Tech Stack:** Python 3.11, pandas, scikit-learn, MLflow (store de fichiers local), FastAPI + Uvicorn, Pydantic v2, joblib, pytest, Docker, gh CLI.

## Global Constraints

- Séparateur CSV source : `;` ; encodage à tester `utf-8` puis repli `latin-1` si échec.
- Niveau géographique retenu : `GEO_OBJECT == "COM"` uniquement.
- Mesure retenue : `DERA_MEASURE == "SALAIRE_NET_EQTP_MENSUEL_MOYENNE"` uniquement.
- Catégories PCS valides : `"1T3"`, `"4"`, `"5"`, `"6"` (jamais `"_T"`).
- Sexes valides : `"F"`, `"M"` (jamais `"_T"`).
- `random_state=42` partout où un générateur aléatoire est utilisé (reproductibilité).
- Split train/test 80/20 **avant** tout traitement des valeurs extrêmes (pas de fuite du test vers le train).
- Le test (`y_test`) n'est jamais borné/modifié ; seul `y_train` est borné aux 1er/99e centiles par `PCS_ESE`.
- MLflow : store de fichiers local (`file:./mlruns`), pas de serveur distant.
- Le conteneur Docker ne ré-entraîne jamais le modèle : il copie `models/best_model.joblib` déjà produit.
- `data/raw/*.csv` et `*.zip` ne sont jamais commités dans git.
- Le repo final est poussé vers un dépôt GitHub **privé** (compte `gh` déjà authentifié : `BISTOUFLEX`).

---

## Task 1: Scaffolding du projet (dépendances, script de données, config)

**Files:**
- Create: `requirements.txt`
- Create: `data/download_data.sh`
- Create: `.dockerignore`
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `src/api/__init__.py`

**Interfaces:**
- Produces: un environnement Python installable (`pip install -r requirements.txt`) et un repo root importable comme package `src` pour tous les tasks suivants.

- [ ] **Step 1: Créer `requirements.txt`**

```text
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
matplotlib==3.9.2
mlflow==2.17.2
fastapi==0.115.4
uvicorn[standard]==0.32.0
pydantic==2.9.2
joblib==1.4.2
pytest==8.3.3
httpx==0.27.2
```

- [ ] **Step 2: Créer `pyproject.toml` pour que pytest trouve le package `src` depuis n'importe quel dossier**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Créer les fichiers `__init__.py` vides**

```bash
touch src/__init__.py src/api/__init__.py
```

- [ ] **Step 4: Créer `data/download_data.sh` (script de ré-téléchargement reproductible du dataset Insee)**

```bash
#!/usr/bin/env bash
set -euo pipefail

URL="https://api.insee.fr/melodi/file/DS_BTS_SAL_EQTP_SEX_PCS/DS_BTS_SAL_EQTP_SEX_PCS_2024_CSV_FR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/raw"
ZIP_PATH="$DEST_DIR/DS_BTS_SAL_EQTP_SEX_PCS_2024_CSV_FR.zip"

mkdir -p "$DEST_DIR"
echo "Téléchargement du dataset Insee (salaires EQTP, secteur privé)..."
curl -sL -o "$ZIP_PATH" "$URL"
unzip -o "$ZIP_PATH" -d "$DEST_DIR"
echo "OK : fichiers extraits dans $DEST_DIR"
```

- [ ] **Step 5: Rendre le script exécutable et vérifier qu'il fonctionne (le fichier existe déjà, mais on valide le script)**

Run: `chmod +x data/download_data.sh && bash data/download_data.sh`
Expected: `OK : fichiers extraits dans .../data/raw` et les fichiers `DS_BTS_SAL_EQTP_SEX_PCS_2024_data.csv` / `..._metadata.csv` présents dans `data/raw/`.

- [ ] **Step 6: Installer les dépendances dans l'environnement du projet**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt`
Expected: installation sans erreur (dernière ligne `Successfully installed ...`).

- [ ] **Step 7: Créer `.dockerignore` vide pour l'instant (rempli à la Task 6, une fois `src/` et `models/` existants)**

Run: `touch .dockerignore`

- [ ] **Step 8: Commit**

```bash
git add requirements.txt pyproject.toml data/download_data.sh src/__init__.py src/api/__init__.py .dockerignore
git commit -m "chore: scaffolding (dépendances, script de données, config pytest)"
```

(Le fichier `.dockerignore` est créé vide à ce stade — il sera rempli à la Task 6.)

---

## Task 2: Module de données (`src/data.py`)

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `code_departement(geo: str) -> str`
  - `load_raw(data_path: str | Path) -> pd.DataFrame`
  - `filter_commune_level(df: pd.DataFrame) -> pd.DataFrame` (ajoute la colonne `DEP`, garde `GEO, GEO_OBJECT, SEX, PCS_ESE, TIME_PERIOD, DEP, OBS_VALUE`)
  - `build_preprocessor() -> sklearn.compose.ColumnTransformer`
  - `clip_outliers_by_pcs(train: pd.DataFrame) -> pd.Series`
  - `prepare_dataset(data_path) -> tuple[X_train, X_test, y_train, y_test]` (DataFrames/Series pandas)
  - Constantes : `FEATURES = ["SEX", "PCS_ESE", "DEP", "TIME_PERIOD"]`, `TARGET = "OBS_VALUE"`, `RANDOM_STATE = 42`
- Consumes: rien (premier module du pipeline).

- [ ] **Step 1: Écrire les tests (ils doivent échouer, `src/data.py` n'existe pas encore)**

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest

from src.data import (
    FEATURES,
    TARGET,
    build_preprocessor,
    clip_outliers_by_pcs,
    code_departement,
    filter_commune_level,
    load_raw,
    prepare_dataset,
)


def _write_raw_csv(path, rows):
    header = "GEO;GEO_OBJECT;FREQ;SEX;PCS_ESE;DERA_MEASURE;OBS_STATUS;CONF_STATUS;TIME_PERIOD;OBS_VALUE\n"
    lines = [header] + [";".join(row) + "\n" for row in rows]
    path.write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def raw_csv(tmp_path):
    rows = [
        # commune, F/M, PCS 1T3/4/5/6, valeurs présentes -> exploitables
        ["01004", "COM", "A", "F", "1T3", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2023", "3367.0"],
        ["01004", "COM", "A", "M", "1T3", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2023", "4069.0"],
        ["01004", "COM", "A", "F", "5", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2023", "1812.0"],
        ["75056", "COM", "A", "M", "6", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2024", "1950.0"],
        ["97400", "COM", "A", "F", "4", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2024", "2200.0"],
        # valeur manquante (secret statistique) -> doit être filtrée
        ["02000", "COM", "A", "F", "6", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "C", "2023", ""],
        # ligne agrégée sexe "_T" -> doit être filtrée
        ["01004", "COM", "A", "_T", "1T3", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2023", "3700.0"],
        # ligne agrégée métier "_T" -> doit être filtrée
        ["01004", "COM", "A", "F", "_T", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2023", "3100.0"],
        # niveau région, pas commune -> doit être filtrée
        ["11", "REG", "A", "F", "1T3", "SALAIRE_NET_EQTP_MENSUEL_MOYENNE", "A", "F", "2023", "4200.0"],
        # autre mesure -> doit être filtrée
        ["01004", "COM", "A", "F", "1T3", "AUTRE_MESURE", "A", "F", "2023", "9999.0"],
    ]
    path = tmp_path / "data.csv"
    _write_raw_csv(path, rows)
    return path


def test_code_departement_metropole():
    assert code_departement("01004") == "01"


def test_code_departement_outremer():
    assert code_departement("97400") == "974"


def test_load_raw_reads_semicolon_separated_file(raw_csv):
    df = load_raw(raw_csv)
    assert list(df.columns) == [
        "GEO", "GEO_OBJECT", "FREQ", "SEX", "PCS_ESE", "DERA_MEASURE",
        "OBS_STATUS", "CONF_STATUS", "TIME_PERIOD", "OBS_VALUE",
    ]
    assert len(df) == 10


def test_filter_commune_level_keeps_only_exploitable_rows(raw_csv):
    raw = load_raw(raw_csv)
    df = filter_commune_level(raw)

    # 5 lignes exploitables attendues (voir fixture)
    assert len(df) == 5
    assert set(df["GEO_OBJECT"].unique()) == {"COM"}
    assert set(df["SEX"].unique()) <= {"F", "M"}
    assert "_T" not in df["PCS_ESE"].unique()
    assert df["OBS_VALUE"].notna().all()


def test_filter_commune_level_adds_department_code(raw_csv):
    raw = load_raw(raw_csv)
    df = filter_commune_level(raw)
    dep_by_geo = dict(zip(df["GEO"], df["DEP"]))
    assert dep_by_geo["01004"] == "01"
    assert dep_by_geo["75056"] == "75"
    assert dep_by_geo["97400"] == "974"


def test_build_preprocessor_has_onehot_and_scaler():
    prep = build_preprocessor()
    names = [name for name, _, _ in prep.transformers]
    assert names == ["cat", "num"]


def test_clip_outliers_by_pcs_bounds_within_train_quantiles():
    train = pd.DataFrame({
        "PCS_ESE": ["4"] * 10,
        "OBS_VALUE": list(range(1, 11)),  # 1..10
    })
    clipped = clip_outliers_by_pcs(train)
    # le 1er et le dernier (valeurs extrêmes) doivent être ramenés à l'intérieur de [q01, q99]
    assert clipped.min() >= train["OBS_VALUE"].quantile(0.01)
    assert clipped.max() <= train["OBS_VALUE"].quantile(0.99)


def test_prepare_dataset_returns_consistent_shapes(raw_csv):
    X_train, X_test, y_train, y_test = prepare_dataset(raw_csv)
    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES
    assert len(X_train) + len(X_test) == 5
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent (module manquant)**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest tests/test_data.py -v`
Expected: `ModuleNotFoundError: No module named 'src.data'` (ou équivalent `ImportError`).

- [ ] **Step 3: Implémenter `src/data.py`**

```python
"""Chargement et préparation du dataset Insee des salaires (réutilisé du module 2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MEASURE = "SALAIRE_NET_EQTP_MENSUEL_MOYENNE"
NIVEAU = "COM"
VAR_CAT = ["SEX", "PCS_ESE", "DEP"]
VAR_NUM = ["TIME_PERIOD"]
FEATURES = VAR_CAT + VAR_NUM
TARGET = "OBS_VALUE"
RANDOM_STATE = 42

PCS_LABELS = {
    "1T3": "Cadres",
    "4": "Prof. intermédiaires",
    "5": "Employés",
    "6": "Ouvriers",
}


def code_departement(geo: str) -> str:
    """Déduit le département des 2 premiers caractères du code commune (3 pour l'outre-mer, préfixe 97)."""
    return geo[:3] if geo.startswith("97") else geo[:2]


def load_raw(data_path: str | Path) -> pd.DataFrame:
    """Charge le CSV Insee brut : séparateur `;`, tout en texte (les codes commencent parfois par 0)."""
    options = dict(sep=";", dtype=str, na_values=["", "NA", "ND", "s"])
    try:
        return pd.read_csv(data_path, encoding="utf-8", **options)
    except UnicodeDecodeError:
        return pd.read_csv(data_path, encoding="latin-1", **options)


def filter_commune_level(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les lignes commune, mesure de salaire EQTP, sexe F/M, métier non agrégé, salaire publié."""
    mask = (
        (df["GEO_OBJECT"] == NIVEAU)
        & (df["DERA_MEASURE"] == MEASURE)
        & (df["SEX"].isin(["F", "M"]))
        & (df["PCS_ESE"] != "_T")
    )
    out = df.loc[mask].copy()
    out["OBS_VALUE"] = pd.to_numeric(out["OBS_VALUE"], errors="coerce")
    out["TIME_PERIOD"] = pd.to_numeric(out["TIME_PERIOD"], errors="coerce")
    out = out.dropna(subset=["OBS_VALUE", "TIME_PERIOD", "GEO"])
    out["TIME_PERIOD"] = out["TIME_PERIOD"].astype(int)
    out["DEP"] = out["GEO"].apply(code_departement)
    return out.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """One-hot pour les catégorielles, standardisation pour l'année."""
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), VAR_CAT),
            ("num", StandardScaler(), VAR_NUM),
        ]
    )


def clip_outliers_by_pcs(train: pd.DataFrame) -> pd.Series:
    """Ramène le salaire d'entraînement aux 1er/99e centiles, calculés par métier, sur le train uniquement."""
    bornes = train.groupby("PCS_ESE")[TARGET].quantile([0.01, 0.99]).unstack()
    bas = train["PCS_ESE"].map(bornes[0.01])
    haut = train["PCS_ESE"].map(bornes[0.99])
    return train[TARGET].clip(bas, haut)


def prepare_dataset(data_path: str | Path):
    """Charge, filtre, split 80/20, borne les extrêmes du train. Retourne X_train, X_test, y_train, y_test."""
    raw = load_raw(data_path)
    df = filter_commune_level(raw)
    train, test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    y_train = clip_outliers_by_pcs(train)
    y_test = test[TARGET]
    return train[FEATURES], test[FEATURES], y_train, y_test
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest tests/test_data.py -v`
Expected: 8 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: module de chargement et préparation des données (src/data.py)"
```

---

## Task 3: Entraînement multi-modèles + suivi MLflow (`src/train.py`)

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `src.data.build_preprocessor`, `src.data.prepare_dataset`, `src.data.FEATURES`
- Produces:
  - `get_model_configs() -> dict[str, tuple[estimator, dict]]`
  - `build_pipeline(model) -> sklearn.pipeline.Pipeline`
  - `train_and_log(name, model, params, X_train, y_train, X_test, y_test, artifacts_dir) -> dict` avec clés `name, run_id, pipeline, test_mae, test_rmse, test_r2, cv_rmse`
  - `select_best(results: list[dict]) -> dict` (choisit la plus petite `test_rmse`, en excluant `name == "reference_moyenne"`)
  - `run_training(data_path, models_dir) -> dict` (model_card), point d'entrée `if __name__ == "__main__"`

- [ ] **Step 1: Écrire les tests (ils doivent échouer, `src/train.py` n'existe pas encore)**

Create `tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest tests/test_train.py -v`
Expected: `ModuleNotFoundError: No module named 'src.train'`.

- [ ] **Step 3: Implémenter `src/train.py`**

```python
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
```

- [ ] **Step 4: Lancer les tests pour vérifier qu'ils passent**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest tests/test_train.py -v`
Expected: 2 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: entraînement multi-modèles avec suivi MLflow (src/train.py)"
```

---

## Task 4: API FastAPI de prédiction (`src/api/`)

**Files:**
- Create: `src/api/schemas.py`
- Create: `src/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: le fichier `models/best_model.joblib` (pipeline sklearn `.predict(DataFrame)`) et `models/model_card.json` (clés `model_name, test_rmse, test_mae`), tous deux produits par `src.train.run_training` (Task 3/5).
- Produces: application FastAPI `src.api.main.app` avec `GET /health` et `POST /predict` ; variable de module `MODELS_DIR` (patchable dans les tests).

- [ ] **Step 1: Écrire les tests (ils doivent échouer, les modules n'existent pas encore)**

Create `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest tests/test_api.py -v`
Expected: `ModuleNotFoundError: No module named 'src.api.main'`.

- [ ] **Step 3: Implémenter `src/api/schemas.py`**

```python
"""Schémas Pydantic pour l'API de prédiction de salaire."""
from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sex: Literal["F", "M"] = Field(..., description="Sexe : F ou M")
    pcs: Literal["1T3", "4", "5", "6"] = Field(
        ...,
        description="Catégorie socioprofessionnelle Insee : 1T3 (cadres), 4 (prof. intermédiaires), 5 (employés), 6 (ouvriers)",
    )
    department: str = Field(..., description="Code département (ex: '75', '2A', '974')")
    year: int = Field(..., ge=2000, le=2100, description="Année")


class PredictionResponse(BaseModel):
    predicted_salary: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    test_rmse: float
    test_mae: float
```

- [ ] **Step 4: Implémenter `src/api/main.py`**

```python
"""API FastAPI de prédiction du salaire net mensuel moyen (EQTP, secteur privé, Insee)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

app = FastAPI(
    title="API de prédiction des salaires (Insee EQTP)",
    description="Prédit le salaire net mensuel moyen à partir du sexe, du métier, du département et de l'année.",
    version="1.0.0",
)

_state: dict = {}


@app.on_event("startup")
def load_model() -> None:
    _state["pipeline"] = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "model_card.json", encoding="utf-8") as f:
        _state["model_card"] = json.load(f)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    card = _state["model_card"]
    return HealthResponse(
        status="ok",
        model_name=card["model_name"],
        test_rmse=card["test_rmse"],
        test_mae=card["test_mae"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    row = pd.DataFrame([{
        "SEX": request.sex,
        "PCS_ESE": request.pcs,
        "DEP": request.department,
        "TIME_PERIOD": request.year,
    }])
    prediction = _state["pipeline"].predict(row)[0]
    return PredictionResponse(
        predicted_salary=round(float(prediction), 2),
        model_used=_state["model_card"]["model_name"],
    )
```

- [ ] **Step 5: Lancer les tests pour vérifier qu'ils passent**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest tests/test_api.py -v`
Expected: 4 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas.py src/api/main.py tests/test_api.py
git commit -m "feat: API FastAPI de prédiction (/health, /predict)"
```

---

## Task 5: Entraînement réel sur le dataset complet

**Files:**
- Modify: `models/best_model.joblib` (créé par exécution, pas édité manuellement)
- Modify: `models/model_card.json` (créé par exécution)
- Modify: `mlruns/` (créé par exécution)

**Interfaces:**
- Consumes: `src.train.run_training` (Task 3), `data/raw/DS_BTS_SAL_EQTP_SEX_PCS_2024_data.csv` (Task 1)
- Produces: les deux artefacts (`models/best_model.joblib`, `models/model_card.json`) consommés par l'API (Task 4) et par le `Dockerfile` (Task 6).

- [ ] **Step 1: Lancer tous les tests unitaires une dernière fois avant l'exécution réelle (non-régression)**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m pytest -v`
Expected: tous les tests `PASSED` (Task 2 + Task 3 + Task 4 cumulés).

- [ ] **Step 2: Lancer l'entraînement réel sur le dataset complet**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m src.train`
Expected : 4 lignes `Entraînement : ...` puis une ligne finale `Meilleur modèle : <nom> (RMSE test = ... €, R² = ...)`. Cette commande peut prendre plusieurs minutes (Random Forest sur ~65 000 lignes).

- [ ] **Step 3: Vérifier les artefacts produits**

Run: `ls models/ && cat models/model_card.json`
Expected: `best_model.joblib` et `model_card.json` présents ; le JSON contient `model_name`, `test_rmse`, `test_mae`, `test_r2`, `tous_les_runs` (4 entrées).

- [ ] **Step 4: Vérifier que l'API démarre avec le vrai modèle et répond correctement**

Run:
```bash
"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m uvicorn src.api.main:app --port 8001 &
sleep 2
curl -s http://127.0.0.1:8001/health
curl -s -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d '{"sex":"F","pcs":"1T3","department":"75","year":2024}'
kill %1
```
Expected: `/health` retourne le nom du vrai modèle sélectionné ; `/predict` retourne un `predicted_salary` numérique plausible (quelques milliers d'euros).

- [ ] **Step 5: Lancer l'UI MLflow pour visualiser et comparer les essais (vérification manuelle, pas de commit)**

Run: `"C:\Users\romai\AppData\Local\Python\bin\python.exe" -m mlflow ui --backend-store-uri file:./mlruns --port 5000`
Expected: ouverture possible sur `http://127.0.0.1:5000`, expérience `salaires-eqtp-prediction` avec 4 runs comparables (colonnes `test_rmse`, `test_mae`, `test_r2`). Arrêter avec Ctrl+C une fois vérifié.

- [ ] **Step 6: Commit des artefacts (modèle + logs MLflow, livrables explicites de l'énoncé)**

```bash
git add models/best_model.joblib models/model_card.json mlruns/
git commit -m "chore: entraînement complet + modèle sélectionné + logs MLflow"
```

---

## Task 6: Dockerfile et test local de l'image

**Files:**
- Create: `Dockerfile`
- Modify: `.dockerignore`

**Interfaces:**
- Consumes: `requirements.txt` (Task 1), `src/` (Tasks 2-4), `models/best_model.joblib` + `models/model_card.json` (Task 5).
- Produces: image Docker `salaires-api:latest` exposant le port 8000 avec les mêmes endpoints que Task 4.

- [ ] **Step 1: Remplir `.dockerignore`**

```text
data/
mlruns/
tests/
docs/
.git/
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/
```

- [ ] **Step 2: Créer `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Construire l'image**

Run: `docker build -t salaires-api .`
Expected: build qui se termine par `Successfully tagged salaires-api:latest` (ou équivalent BuildKit `naming to docker.io/library/salaires-api:latest done`).

- [ ] **Step 4: Lancer le conteneur**

Run: `docker run -d --name salaires-api-test -p 8000:8000 salaires-api`
Expected: un ID de conteneur est affiché, pas d'erreur.

- [ ] **Step 5: Tester les endpoints depuis l'hôte**

Run:
```bash
sleep 2
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"sex":"M","pcs":"6","department":"69","year":2023}'
```
Expected: `/health` renvoie `{"status":"ok", ...}` ; `/predict` renvoie un JSON avec `predicted_salary` et `model_used`.

- [ ] **Step 6: Arrêter et nettoyer le conteneur de test**

Run: `docker stop salaires-api-test && docker rm salaires-api-test`
Expected: le nom du conteneur est affiché deux fois (stop puis rm), pas d'erreur.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: Dockerfile de l'API (image testée en local avec docker build/run)"
```

---

## Task 7: README pédagogique et commit final

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: les métriques réelles produites à la Task 5 (`models/model_card.json`) pour renseigner le tableau de résultats.

- [ ] **Step 1: Lire les métriques réelles pour les reporter dans le README**

Run: `cat models/model_card.json`
(Noter le `model_name` gagnant et les `test_mae/test_rmse/test_r2` de chacun des 4 runs listés dans `tous_les_runs` pour remplir le tableau ci-dessous.)

- [ ] **Step 2: Créer `README.md`**

```markdown
# Prédiction de salaires — Atelier pratique 3 (Module 3 MLOps)

Pipeline complet : données Insee (module 2) → entraînement multi-modèles avec suivi MLflow → API FastAPI → image Docker.

## Dataset

Salaires nets mensuels moyens EQTP, secteur privé, par commune/sexe/métier (Insee, via data.gouv.fr).
Réutilisé depuis le module 2. Pour le retélécharger :

\`\`\`bash
bash data/download_data.sh
\`\`\`

Niveau retenu : commune (`GEO_OBJECT == "COM"`). Features : sexe, catégorie socioprofessionnelle
(`1T3`=cadres, `4`=prof. intermédiaires, `5`=employés, `6`=ouvriers), département, année.
Seules ~14,5 % des lignes commune ont un salaire publié (secret statistique Insee sur les petits effectifs).

## Installation

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Entraînement et suivi MLflow

\`\`\`bash
python -m src.train
\`\`\`

Entraîne 4 modèles (référence, régression linéaire, Random Forest, Gradient Boosting), logue chaque essai
dans MLflow (`./mlruns`, store local), sélectionne automatiquement le meilleur (RMSE test la plus basse)
et l'exporte en `models/best_model.joblib`.

Pour comparer visuellement les essais :

\`\`\`bash
mlflow ui --backend-store-uri file:./mlruns
\`\`\`

### Résultats (dernier entraînement)

| Modèle | MAE test (€) | RMSE test (€) | R² test |
|---|---|---|---|
<!-- une ligne par entrée de model_card.json -> tous_les_runs, remplie à l'étape 1 -->

## API

\`\`\`bash
uvicorn src.api.main:app --reload
\`\`\`

- `GET /health` : statut + modèle chargé + métriques d'entraînement.
- `POST /predict` : `{"sex": "F", "pcs": "1T3", "department": "75", "year": 2024}` → `{"predicted_salary": ..., "model_used": "..."}`.

## Docker

\`\`\`bash
docker build -t salaires-api .
docker run -p 8000:8000 salaires-api
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sex":"F","pcs":"1T3","department":"75","year":2024}'
\`\`\`

## Tests

\`\`\`bash
pytest -v
\`\`\`

## Limites connues

- Une catégorie non vue à l'entraînement (ex. département absent) est encodée à zéro par le `OneHotEncoder`
  (`handle_unknown="ignore"`) : la prédiction reste possible mais moins fiable.
- Pas de recherche d'hyperparamètres (`GridSearchCV`) : hyperparamètres fixes, choisis pour rester rapides
  à exécuter. Le notebook du module 2 contient la recherche complète si besoin de l'approfondir.
- Secret statistique Insee : les petites communes sont sous-représentées dans les données exploitables.
```

- [ ] **Step 3: Remplacer le commentaire du tableau par les vraies lignes issues de `model_card.json`**

(Étape manuelle : copier les 4 valeurs de `tous_les_runs` dans le tableau Markdown, remplacer `reference_moyenne`/`regression_lineaire`/`random_forest`/`gradient_boosting` par des libellés lisibles : Référence (moyenne), Régression linéaire, Random Forest, Gradient Boosting.)

- [ ] **Step 4: Commit final**

```bash
git add README.md
git commit -m "docs: README pédagogique (installation, entraînement, API, Docker, résultats)"
```

---

## Task 8: Création du repo GitHub privé et push

**Files:** aucun fichier modifié — opération git/GitHub uniquement.

**Interfaces:** aucune (dernière étape, après validation explicite de l'utilisateur).

- [ ] **Step 1: Vérifier l'état du repo avant de pousser (rien d'inattendu en attente)**

Run: `git status && git log --oneline`
Expected: working tree clean, historique des commits des Tasks 1 à 7.

- [ ] **Step 2: Demander confirmation explicite à l'utilisateur avant de créer un repo et de pousser du code (action visible/externe)**

(Ne pas exécuter l'étape 3 sans un "oui" explicite dans la conversation — c'est une action qui publie du code vers un service externe.)

- [ ] **Step 3: Créer le repo GitHub privé et pousser**

Run:
```bash
gh repo create atelier3-salaires-mlops --private --source=. --remote=origin --push
```
Expected: sortie `gh` confirmant la création du repo privé sous le compte `BISTOUFLEX` et un push réussi de `master`/`main`.

- [ ] **Step 4: Vérifier le résultat**

Run: `gh repo view --web`
(Ou partager l'URL affichée par la commande précédente avec l'utilisateur.)
