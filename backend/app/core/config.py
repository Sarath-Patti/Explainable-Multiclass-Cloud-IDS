"""Core configuration settings for the Explainable Multiclass Cloud IDS FastAPI backend."""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project directory (Explainable-Multiclass-Cloud-IDS)
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings and environment variable bindings."""

    PROJECT_NAME: str = "Explainable Multiclass Cloud IDS API"
    VERSION: str = "1.1"
    API_V1_STR: str = "/api/v1"

    # Artifact paths
    MODELS_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = BASE_DIR / "models" / "xgboost_shap_selected.pkl"
    LABEL_ENCODER_PATH: Path = BASE_DIR / "models" / "label_encoder.pkl"
    LABEL_MAPPING_PATH: Path = BASE_DIR / "models" / "label_mapping.json"

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
