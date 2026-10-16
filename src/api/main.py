"""API FastAPI de prédiction du salaire net mensuel moyen (EQTP, secteur privé, Insee)."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = joblib.load(MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "model_card.json", encoding="utf-8") as f:
        _state["model_card"] = json.load(f)
    yield
    _state.clear()


app = FastAPI(
    title="API de prédiction des salaires (Insee EQTP)",
    description="Prédit le salaire net mensuel moyen à partir du sexe, du métier, du département et de l'année.",
    version="1.0.0",
    lifespan=lifespan,
)


app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirige vers l'interface de démo, pour éviter un 404 quand on visite juste le domaine."""
    return RedirectResponse(url="/ui")


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
