"""FastAPI application entrypoint for Explainable Multiclass Cloud IDS."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.services.model_loader import model_loader

logger = logging.getLogger("ApplicationMain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Initializing application startup sequence...")
    try:
        model_loader.load_artifacts()
        logger.info("ML model and label mapping binaries initialized successfully.")
    except Exception as e:
        logger.error(f"Startup warning: ML artifacts could not be initialized: {str(e)}")
    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    description="REST API backend for Explainable Multiclass Cloud Intrusion Detection System.",
    lifespan=lifespan,
)

# Set up CORS middleware
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include v1 API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Root endpoint info and docs pointer."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": f"{settings.API_V1_STR}/docs",
        "health": f"{settings.API_V1_STR}/health",
        "predict": f"{settings.API_V1_STR}/predict",
    }
