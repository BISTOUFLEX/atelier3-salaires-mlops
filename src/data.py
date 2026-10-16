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
