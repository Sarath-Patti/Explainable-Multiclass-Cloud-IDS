"""Prediction service executing batch ML model inference on network flow DataFrames."""

import logging
from typing import List
import numpy as np
import pandas as pd
from app.services.model_loader import model_loader
from app.schemas.predict import PredictionItem, PredictionSummary, PredictionResponse

logger = logging.getLogger("PredictorService")


class PredictionError(Exception):
    """Base exception for prediction service errors."""
    pass


class EmptyDataError(PredictionError):
    """Raised when the input DataFrame contains zero rows."""
    pass


class MissingFeaturesError(PredictionError):
    """Raised when one or more required model features are missing from input DataFrame."""

    def __init__(self, missing_features: List[str]):
        self.missing_features = missing_features
        super().__init__(f"Missing required feature(s): {', '.join(missing_features)}")


class PredictionService:
    """Service to handle DataFrame feature validation, model inference, and output formatting."""

    @staticmethod
    def predict_dataframe(df: pd.DataFrame) -> PredictionResponse:
        """Performs batch multiclass classification on input network flow DataFrame.

        Args:
            df: Input pandas DataFrame containing flow parameters.

        Returns:
            PredictionResponse containing summary stats and per-row prediction objects.
        """
        if df.empty or len(df) == 0:
            raise EmptyDataError("Input dataset contains no rows for prediction.")

        expected_features = model_loader.expected_features
        model = model_loader.model
        index_to_label = model_loader.index_to_label

        # 1. Validate that all required features exist in input DataFrame
        missing = [f for f in expected_features if f not in df.columns]
        if missing:
            logger.warning(f"Prediction failed: missing {len(missing)} required features: {missing}")
            raise MissingFeaturesError(missing)

        # 2. Slice DataFrame using exact expected feature order
        X = df.loc[:, expected_features].copy()

        # 3. Clean numeric values (replace inf / -inf with NaN, fill missing with 0.0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 4. Perform batch model inference
        logger.info(f"Executing batch inference on {len(X)} records with {len(expected_features)} features...")
        try:
            probabilities = model.predict_proba(X)
        except Exception as e:
            logger.exception("Error executing model predict_proba:")
            raise PredictionError(f"Model execution error: {str(e)}") from e

        pred_indices = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        # 5. Format predictions and compute summary statistics (vectorized)
        labels = [index_to_label.get(int(idx), f"Unknown_{idx}") for idx in pred_indices]
        rounded_confidences = np.round(confidences, 4)

        benign_count = sum(1 for lbl in labels if lbl.upper() == "BENIGN")

        prediction_items = [
            PredictionItem(
                row=i,
                prediction=labels[i],
                confidence=float(rounded_confidences[i])
            )
            for i in range(len(X))
        ]

        total_samples = len(X)
        attack_count = total_samples - benign_count

        summary = PredictionSummary(
            total_samples=total_samples,
            predicted_attacks=attack_count,
            predicted_benign=benign_count
        )

        logger.info(f"Batch prediction complete: {total_samples} samples evaluated ({attack_count} attacks, {benign_count} benign).")
        return PredictionResponse(summary=summary, predictions=prediction_items)
