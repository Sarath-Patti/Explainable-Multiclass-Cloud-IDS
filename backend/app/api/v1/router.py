"""FastAPI v1 API Router aggregating endpoint routes."""

from fastapi import APIRouter
from app.api.v1.endpoints import health, predict, explain

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(predict.router, tags=["Prediction"])
api_router.include_router(explain.router, tags=["Explainability"])
