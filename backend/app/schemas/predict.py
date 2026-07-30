"""Pydantic schemas for batch inference request and response payloads."""

from typing import List
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Individual row prediction result."""

    row: int = Field(..., example=0, description="Zero-indexed row number from input CSV")
    prediction: str = Field(..., example="BENIGN", description="Predicted security class label")
    confidence: float = Field(..., example=0.9984, description="Prediction probability confidence score (0.0 to 1.0)")


class PredictionSummary(BaseModel):
    """Aggregate statistics for batch prediction job."""

    total_samples: int = Field(..., example=1000, description="Total number of evaluated flow samples")
    predicted_attacks: int = Field(..., example=213, description="Total number of samples classified as attack traffic")
    predicted_benign: int = Field(..., example=787, description="Total number of samples classified as BENIGN traffic")


class PredictionResponse(BaseModel):
    """Response payload returned by POST /api/v1/predict."""

    summary: PredictionSummary
    predictions: List[PredictionItem]
