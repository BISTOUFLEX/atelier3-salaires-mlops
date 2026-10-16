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
