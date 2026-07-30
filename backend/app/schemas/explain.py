"""Pydantic schemas for SHAP explainability request and response payloads."""

from typing import Dict, List, Any
from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Individual feature attribution object."""

    feature: str = Field(..., example="Destination Port", description="Name of the flow feature")
    value: float = Field(..., example=80.0, description="Observed raw numeric feature value")
    shap_value: float = Field(..., example=4.82, description="SHAP attribution value towards predicted class")


class ExplainRequest(BaseModel):
    """Payload sent to POST /api/v1/explain."""

    row: int = Field(0, example=0, description="Zero-indexed row number from batch CSV dataset")
    features: Dict[str, Any] = Field(
        ...,
        example={"Destination Port": 80, "Init_Win_bytes_forward": 29200},
        description="Key-value dictionary mapping feature names to raw flow values"
    )


class ExplainResponse(BaseModel):
    """Response payload returned by POST /api/v1/explain."""

    prediction: str = Field(..., example="DDoS", description="Predicted multiclass security label")
    confidence: float = Field(..., example=0.9984, description="Model prediction confidence (0.0 to 1.0)")
    base_value: float = Field(..., example=0.05, description="Expected base output value (prior probability)")
    top_features: List[FeatureContribution] = Field(
        ...,
        description="List of feature attributions sorted by descending absolute SHAP magnitude"
    )
