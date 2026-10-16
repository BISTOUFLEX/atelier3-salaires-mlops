# Atelier pratique 3 — Pipeline ML + MLflow + Docker + API — Design

Date : 2026-10-16
Contexte : Module 3 "Modélisation avancée & MLOps" (Epitech MSc, Data Business Intelligence). Dataset réutilisé depuis le module 2 (notebook `analyse_salaires.ipynb`) : salaires nets mensuels EQTP dans le secteur privé, par commune, sexe et catégorie socioprofessionnelle (Insee, via data.gouv.fr).

## Objectifs de l'atelier (énoncé, p.14)

1. Préparer le projet ML : charger le dataset du module 2, entraîner un modèle supervisé, sauvegarder le modèle (pickle/joblib).
2. Suivi des expériences avec MLflow : enregistrer paramètres/métriques/artefacts, comparer plusieurs essais, choisir le meilleur modèle.
3. Construire une image Docker : Dockerfile avec code, API (FastAPI), dépendances et modèle ; tester `docker build` + `docker run` localement.

Livrables : un repository avec code ML, API, Dockerfile et logs MLflow ; une application dockerisée permettant de faire des prédictions avec suivi des essais.

## Dataset réel (vérifié, pas supposé)

Le notebook référence des fichiers introuvables localement (`/Users/remypicard/Downloads/...`). Retéléchargé depuis la source officielle : data.gouv.fr → Insee (`DS_BTS_SAL_EQTP_SEX_PCS`), édition 2024 (millésimes 2023 et 2024 inclus), ZIP de 5,36 Mo contenant :
- `DS_BTS_SAL_EQTP_SEX_PCS_2024_data.csv` (~98 Mo, ~1,25M lignes, séparateur `;`, colonnes entre guillemets)
- `DS_BTS_SAL_EQTP_SEX_PCS_2024_metadata.csv` (dictionnaire des codes)

Colonnes : `GEO`, `GEO_OBJECT`, `FREQ`, `SEX`, `PCS_ESE`, `DERA_MEASURE`, `OBS_STATUS`, `CONF_STATUS`, `TIME_PERIOD`, `OBS_VALUE`.

Filtrage retenu (niveau commune) :
- `GEO_OBJECT == "COM"` (1 046 250 lignes)
- `SEX in {"F", "M"}` (on exclut `_T`, l'agrégat)
- `PCS_ESE != "_T"` — catégories présentes : `1T3` (cadres, fusion 1/2/3), `4` (prof. intermédiaires), `5` (employés), `6` (ouvriers)
- `OBS_VALUE` non manquant — seulement **14,5 %** des lignes commune ont un salaire publié (secret statistique Insee sur les petits effectifs), soit **~81 000 lignes exploitables**

Différence notable avec le notebook module 2 : la catégorie "cadres" est codée `1T3` et non `3` (nomenclature Insee mise à jour). Deux années disponibles (2023, 2024) au lieu d'une seule.

## Architecture retenue

```
atelier3-salaires-mlops/
├── data/
│   ├── raw/                  # CSV Insee téléchargés (non commités, voir .gitignore)
│   └── download_data.sh      # script de ré-téléchargement reproductible
├── src/
│   ├── data.py                # chargement + filtrage + feature engineering (partagé train/API)
│   ├── train.py               # entraînement multi-modèles + tracking MLflow + sélection + export
│   └── api/
│       ├── main.py            # app FastAPI (endpoints /predict, /health)
│       └── schemas.py         # modèles Pydantic (requête/réponse)
├── models/
│   └── best_model.joblib      # pipeline sklearn complet (prétraitement + modèle), commité (livrable)
├── mlruns/                     # store MLflow local (logs des essais), commité (livrable)
├── tests/
│   └── test_api.py            # test de fumée de l'API (health + predict)
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

### 1. Pipeline de données (`src/data.py`)

- Lecture des deux CSV (`sep=";"`, tout en `dtype=str` puis conversion ciblée), fusion non nécessaire côté metadata pour l'inférence (seulement pour la doc).
- Filtrage décrit ci-dessus.
- Feature engineering : `DEP` déduit des 2 premiers caractères du code commune (3 pour les DOM `97xxx`), `TIME_PERIOD` en entier.
- Variables d'entrée (`FEATURES`) : `SEX`, `PCS_ESE`, `DEP` (catégorielles) + `TIME_PERIOD` (numérique). Cible : `OBS_VALUE`.
- Split train/test 80/20 (`random_state` fixe) **avant** tout traitement des valeurs extrêmes, pour éviter toute fuite d'information.
- Bornage des valeurs extrêmes du salaire d'entraînement au 1er/99e centile, calculé par catégorie socioprofessionnelle, sur le train uniquement (le test n'est jamais modifié) — même logique que le notebook module 2.
- Préprocesseur `ColumnTransformer` : `OneHotEncoder(handle_unknown="ignore")` sur les catégorielles + `StandardScaler` sur l'année.

### 2. Entraînement & comparaison (`src/train.py`)

Quatre pipelines (prétraitement + modèle), hyperparamètres fixes et raisonnables (pas de `GridSearchCV`, pour rester rapide à exécuter dans le temps imparti) :

| Modèle | Rôle | Hyperparamètres |
|---|---|---|
| `DummyRegressor(strategy="mean")` | Référence | — |
| `LinearRegression` | Baseline interprétable | — |
| `RandomForestRegressor` | Capture les effets croisés | `n_estimators=150, max_depth=12, min_samples_leaf=10` |
| `HistGradientBoostingRegressor` | Souvent le plus précis | `learning_rate=0.1, max_iter=200, max_leaf_nodes=31` |

Pour chaque modèle, un run MLflow (store local `file:./mlruns`, pas de serveur à administrer) :
- `log_param` : nom du modèle + hyperparamètres
- Entraînement, prédiction sur le test
- `log_metric` : MAE, RMSE, R² (test) + RMSE moyenne en validation croisée 5 plis (sur un échantillon, comme le notebook)
- `log_artifact` : graphique prédit-vs-réel (PNG)
- `mlflow.sklearn.log_model` : le pipeline complet

À la fin : sélection automatique du run avec la meilleure RMSE test (hors référence), export du pipeline correspondant en `models/best_model.joblib` + un `models/model_card.json` (nom du modèle, métriques, catégories vues à l'entraînement — utile à l'API pour documenter ses limites).

Comparaison visuelle possible via `mlflow ui --backend-store-uri ./mlruns`.

### 3. API de prédiction (`src/api/main.py`, FastAPI)

- Chargement de `models/best_model.joblib` au démarrage (évènement `startup`).
- `GET /health` → statut, nom du modèle chargé, métriques d'entraînement.
- `POST /predict` → entrée validée par Pydantic (`sex: Literal["F","M"]`, `pcs: Literal["1T3","4","5","6"]`, `department: str`, `year: int`), sortie `{predicted_salary, model_used}`.
- Limite documentée : une catégorie non vue à l'entraînement (ex. département absent) est encodée à zéro par le `OneHotEncoder` (`handle_unknown="ignore"`) — la prédiction reste possible mais moins fiable, comme dans le notebook.

### 4. Docker

- Modèle entraîné **en amont**, hors conteneur (`python -m src.train`) : le conteneur ne fait que du service, pas de ré-entraînement (build plus rapide, cohérent avec l'énoncé qui demande d'inclure "le modèle" dans l'image).
- `Dockerfile` : `python:3.11-slim`, installation de `requirements.txt`, copie de `src/` et `models/`, exposition du port 8000, démarrage `uvicorn src.api.main:app`.
- Test local : `docker build -t salaires-api .` puis `docker run -p 8000:8000 salaires-api`, puis `curl` sur `/health` et `/predict`.

### 5. Gestion du repository

- `git init` local, commits par étape logique.
- `data/raw/*.csv` et `*.zip` exclus du suivi git (volumineux, reconstructibles via `data/download_data.sh`) — bonne pratique de gouvernance des données.
- `models/best_model.joblib` et `mlruns/` **commités** : ce sont les livrables explicitement demandés par l'énoncé ("un repository avec [...] logs MLflow").
- À la fin, après validation du contenu : création d'un **repository GitHub privé** (compte déjà authentifié via `gh`) et push — décision prise avec l'utilisateur pour un rendu "client" professionnel (lien partageable, historique de commits visible).

## Hors périmètre (assumé, pour tenir le délai du jour)

- Pas de `GridSearchCV` / recherche d'hyperparamètres extensive (le notebook module 2 le fait déjà ; ici on fixe des valeurs raisonnables pour prioriser le pipeline MLOps complet).
- Pas de serveur MLflow distant ni de Model Registry en production — store de fichiers local, suffisant pour l'exercice.
- Pas de CI/CD ni de tests exhaustifs — un test de fumée sur l'API suffit pour valider le livrable.
- Pas de dataset commité dans git (volumineux) — script de téléchargement à la place.
