"""Core configuration settings for the Explainable Multiclass Cloud IDS FastAPI backend."""

import json
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project directory (Explainable-Multiclass-Cloud-IDS)
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings and environment variable bindings."""

    PROJECT_NAME: str = "Explainable Multiclass Cloud IDS API"
    VERSION: str = "1.4"
    API_V1_STR: str = "/api/v1"

    # Artifact paths
    MODELS_DIR: Path = BASE_DIR / "models"
    MODEL_PATH: Path = BASE_DIR / "models" / "xgboost_shap_selected.pkl"
    LABEL_ENCODER_PATH: Path = BASE_DIR / "models" / "label_encoder.pkl"
    LABEL_MAPPING_PATH: Path = BASE_DIR / "models" / "label_mapping.json"

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:80",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()
