"""Pydantic schema for health status check endpoint response."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., example="healthy", description="Operational status of the API service")
    service: str = Field(..., example="Explainable Multiclass Cloud IDS API", description="Name of the service")
    version: str = Field(..., example="1.0", description="Semantic version of the application")
    timestamp: str = Field(..., example="2026-07-30T12:00:00Z", description="ISO 8601 server timestamp")
