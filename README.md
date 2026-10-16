# Prédiction de salaires — Atelier pratique 3 (Module 3 MLOps)

Pipeline complet : données Insee (module 2) → entraînement multi-modèles avec suivi MLflow → API FastAPI → image Docker.

## Démarrage rapide (sur n'importe quelle machine)

Le modèle entraîné (`models/best_model.joblib`) et les logs MLflow (`mlruns/`) sont déjà commités dans le
repo : pas besoin d'installer Python ni de ré-entraîner quoi que ce soit pour faire la démo.

**Seul prérequis : Docker Desktop installé et lancé.**

```bash
git clone https://github.com/BISTOUFLEX/atelier3-salaires-mlops.git
cd atelier3-salaires-mlops
docker compose up --build
```

Puis ouvrir :
- `http://localhost:8000/docs` — API de prédiction (documentation interactive, testable au clic)
- `http://localhost:5000` — MLflow (comparaison des 4 essais : référence, régression linéaire, Random Forest, Gradient Boosting)

Les deux conteneurs (`atelier3-salaires-mlops-api-1` et `atelier3-salaires-mlops-mlflow-1`) sont aussi
visibles et pilotables (start/stop/logs) depuis Docker Desktop. Pour tout arrêter : `docker compose down`.

Les sections suivantes détaillent chaque brique (données, entraînement, API) pour qui veut lire/modifier
le code plutôt que simplement le faire tourner.

## Dataset

Salaires nets mensuels moyens EQTP, secteur privé, par commune/sexe/métier (Insee, via data.gouv.fr).
Réutilisé depuis le module 2. Pour le retélécharger :

```bash
bash data/download_data.sh
```

Niveau retenu : commune (`GEO_OBJECT == "COM"`). Features : sexe, catégorie socioprofessionnelle
(`1T3`=cadres, `4`=prof. intermédiaires, `5`=employés, `6`=ouvriers), département, année.
Seules ~14,5 % des lignes commune ont un salaire publié (secret statistique Insee sur les petits effectifs),
soit environ 81 000 lignes exploitables.

## Installation (pour développer/modifier le code — pas nécessaire pour la démo Docker ci-dessus)

Projet testé avec Python 3.12 (Python 3.14 pose problème : pas encore de wheels précompilées pour pandas/numpy).

```bash
python -m venv .venv
.venv/Scripts/activate   # ou source .venv/bin/activate sur Linux/macOS
pip install -r requirements.txt
```

## Entraînement et suivi MLflow

```bash
python -m src.train
```

Entraîne 4 modèles (référence, régression linéaire, Random Forest, Gradient Boosting), logue chaque essai
dans MLflow (`./mlruns`, store local), sélectionne automatiquement le meilleur (RMSE test la plus basse)
et l'exporte en `models/best_model.joblib`.

Pour comparer visuellement les essais :

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### Résultats (dernier entraînement, dataset complet)

| Modèle | MAE test (€) | RMSE test (€) | R² test |
|---|---|---|---|
| Référence (moyenne) | 664,9 | 885,9 | 0,00 |
| Régression linéaire | 239,9 | 464,2 | 0,73 |
| Random Forest | 221,7 | 454,9 | 0,74 |
| **Gradient Boosting** (sélectionné) | **216,6** | **449,7** | **0,74** |

Le Gradient Boosting est retenu : il réduit l'erreur RMSE de 49 % par rapport à la référence et reste
légèrement meilleur que la régression linéaire et la forêt aléatoire.

## API

```bash
uvicorn src.api.main:app --reload
```

- `GET /health` : statut + modèle chargé + métriques d'entraînement.
- `POST /predict` : `{"sex": "F", "pcs": "1T3", "department": "75", "year": 2024}` → `{"predicted_salary": ..., "model_used": "..."}`.

## Docker

Image seule (API uniquement) :

```bash
docker build -t salaires-api .
docker run -p 8000:8000 salaires-api
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sex":"F","pcs":"1T3","department":"75","year":2024}'
```

### Docker Compose (API + MLflow, tout dans des conteneurs)

```bash
docker compose up --build
```

Démarre deux conteneurs à partir de la **même image** (celle du `Dockerfile`, qui contient déjà tout :
code, dépendances, modèle) :

- `api` : sert les prédictions sur `http://localhost:8000` (doc interactive sur `/docs`).
- `mlflow` : lance `mlflow ui` sur `http://localhost:5000`, avec `./mlruns` monté en volume pour afficher
  les vrais essais déjà entraînés (c'est pourquoi ce service a besoin d'un volume : le dossier `mlruns/`
  n'est pas copié dans l'image, voir `.dockerignore`, pour ne pas figer les logs au moment du build).

Les deux conteneurs apparaissent dans Docker Desktop (projet `atelier3-salaires-mlops`) et peuvent être
démarrés/arrêtés par clic, sans ligne de commande ni installation Python locale.

## Tests

```bash
pytest -v
```

14 tests (module de données, entraînement/MLflow, API) — tous verts.

## Architecture

```
atelier3-salaires-mlops/
├── data/
│   ├── raw/                  # CSV Insee (non commités, volumineux)
│   └── download_data.sh      # script de ré-téléchargement reproductible
├── src/
│   ├── data.py                # chargement + filtrage + feature engineering (partagé train/API)
│   ├── train.py                # entraînement multi-modèles + tracking MLflow + sélection + export
│   └── api/
│       ├── main.py             # app FastAPI (endpoints /predict, /health)
│       └── schemas.py          # modèles Pydantic (requête/réponse)
├── models/
│   ├── best_model.joblib       # pipeline sklearn complet (prétraitement + modèle)
│   └── model_card.json         # métadonnées du modèle sélectionné (métriques, catégories vues)
├── mlruns/                     # store MLflow local (logs des 4 essais)
├── tests/                      # tests pytest (data, train, api)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml           # api + mlflow, lancés comme conteneurs (cf. section Docker)
└── docs/superpowers/           # design et plan d'implémentation détaillés
```

## Limites connues

- Une catégorie non vue à l'entraînement (ex. département absent) est encodée à zéro par le `OneHotEncoder`
  (`handle_unknown="ignore"`) : la prédiction reste possible mais moins fiable.
- Pas de recherche d'hyperparamètres (`GridSearchCV`) : hyperparamètres fixes, choisis pour rester rapides
  à exécuter. Le notebook du module 2 contient la recherche complète si besoin de l'approfondir.
- Secret statistique Insee : les petites communes sont sous-représentées dans les données exploitables.
- Store MLflow local uniquement (pas de serveur de tracking distant ni de Model Registry en production).
