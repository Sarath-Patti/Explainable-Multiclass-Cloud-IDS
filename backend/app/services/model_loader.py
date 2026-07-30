"""Singleton model loader service for deserializing ML model and label mappings."""

import json
import logging
from typing import Dict, List, Optional
import joblib
from xgboost import XGBClassifier
from app.core.config import settings

logger = logging.getLogger("ModelLoader")


class ModelLoader:
    """Singleton service to load and cache trained ML model and label mapping artifacts."""

    _instance: Optional["ModelLoader"] = None
    _model: Optional[XGBClassifier] = None
    _expected_features: Optional[List[str]] = None
    _index_to_label: Optional[Dict[int, str]] = None

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def load_artifacts(self) -> None:
        """Loads trained XGBoost model binary and label mapping from disk once during startup."""
        if self._model is not None:
            logger.info("Model artifacts already loaded and cached.")
            return

        logger.info(f"Loading trained XGBoost model from: {settings.MODEL_PATH}")
        if not settings.MODEL_PATH.exists():
            raise FileNotFoundError(f"Trained model binary not found at: {settings.MODEL_PATH}")

        try:
            self._model = joblib.load(settings.MODEL_PATH)
            booster = self._model.get_booster()
            self._expected_features = booster.feature_names
            if not self._expected_features:
                raise ValueError("Booster does not contain valid feature names.")
            logger.info(f"Loaded model with {len(self._expected_features)} expected features: {self._expected_features}")
        except Exception as e:
            logger.exception("Failed to load trained XGBoost model binary:")
            raise RuntimeError(f"Model loading failure: {str(e)}") from e

        # Load label mapping dictionary
        logger.info(f"Loading label mapping from: {settings.LABEL_MAPPING_PATH}")
        if settings.LABEL_MAPPING_PATH.exists():
            with open(settings.LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
                label_mapping = json.load(f)
            # Invert mapping from {"BENIGN": 0, "Bot": 1} to {0: "BENIGN", 1: "Bot"}
            self._index_to_label = {int(v): str(k) for k, v in label_mapping.items()}
        elif settings.LABEL_ENCODER_PATH.exists():
            le = joblib.load(settings.LABEL_ENCODER_PATH)
            self._index_to_label = {i: str(c) for i, c in enumerate(le.classes_)}
        else:
            raise FileNotFoundError(f"Label mapping artifact not found at {settings.LABEL_MAPPING_PATH} or {settings.LABEL_ENCODER_PATH}")

        logger.info(f"Loaded label mapping for {len(self._index_to_label)} target classes.")

    @property
    def model(self) -> XGBClassifier:
        if self._model is None:
            self.load_artifacts()
        return self._model

    @property
    def expected_features(self) -> List[str]:
        if self._expected_features is None:
            self.load_artifacts()
        return self._expected_features

    @property
    def index_to_label(self) -> Dict[int, str]:
        if self._index_to_label is None:
            self.load_artifacts()
        return self._index_to_label


model_loader = ModelLoader()
