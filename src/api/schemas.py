"""Schémas Pydantic pour l'API de prédiction de salaire."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    sex: Literal["F", "M"] = Field(..., description="Sexe : F ou M")
    pcs: Literal["1T3", "4", "5", "6"] = Field(
        ...,
        description="Catégorie socioprofessionnelle Insee : 1T3 (cadres), 4 (prof. intermédiaires), 5 (employés), 6 (ouvriers)",
    )
    department: str = Field(..., description="Code département (ex: '75', '2A', '974')")
    year: int = Field(..., ge=2000, le=2100, description="Année")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_salary: float
    model_used: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    test_rmse: float
    test_mae: float
