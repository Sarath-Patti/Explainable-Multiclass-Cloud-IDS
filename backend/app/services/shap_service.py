"""SHAP explainability service computing TreeExplainer feature attributions for model predictions."""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import shap
from app.services.model_loader import model_loader
from app.schemas.explain import FeatureContribution, ExplainResponse

logger = logging.getLogger("SHAPService")


class SHAPService:
    """Singleton service to compute local SHAP feature attributions for model predictions."""

    _instance: Optional["SHAPService"] = None
    _explainer: Optional[shap.TreeExplainer] = None

    def __new__(cls) -> "SHAPService":
        if cls._instance is None:
            cls._instance = super(SHAPService, cls).__new__(cls)
        return cls._instance

    def get_explainer(self) -> shap.TreeExplainer:
        """Lazily initializes and returns the SHAP TreeExplainer instance."""
        if self._explainer is None:
            logger.info("Initializing SHAP TreeExplainer for trained XGBoost model...")
            model = model_loader.model
            # Pass model booster or classifier directly
            try:
                self._explainer = shap.TreeExplainer(model)
                logger.info("SHAP TreeExplainer successfully initialized.")
            except Exception as e:
                logger.exception("Failed to initialize SHAP TreeExplainer:")
                raise RuntimeError(f"SHAP explainer initialization error: {str(e)}") from e
        return self._explainer

    def explain_instance(self, features: Dict[str, Any], row_index: int = 0) -> ExplainResponse:
        """Computes local SHAP attributions for a single network flow instance.

        Args:
            features: Key-value dictionary containing feature values.
            row_index: Optional row index identifier.

        Returns:
            ExplainResponse object containing predicted class, confidence, base_value, and sorted attributions.
        """
        expected_features = model_loader.expected_features
        model = model_loader.model
        index_to_label = model_loader.index_to_label
        explainer = self.get_explainer()

        # 1. Create single-row DataFrame and align features
        df = pd.DataFrame([features])

        # Fill any missing required feature columns with 0.0
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0.0

        # Reorder columns strictly according to expected feature order
        X = df.loc[:, expected_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 2. Compute model prediction and confidence
        try:
            probabilities = model.predict_proba(X)[0]
        except Exception as e:
            logger.exception("Error evaluating predict_proba for SHAP explanation:")
            raise RuntimeError(f"Prediction error during SHAP calculation: {str(e)}") from e

        pred_class_idx = int(np.argmax(probabilities))
        confidence = float(round(probabilities[pred_class_idx], 4))
        predicted_label = index_to_label.get(pred_class_idx, f"Class_{pred_class_idx}")

        # 3. Compute SHAP values using TreeExplainer
        logger.info(f"Computing SHAP values for row {row_index} (predicted class: {predicted_label})...")
        try:
            raw_shap = explainer.shap_values(X)
        except Exception as e:
            logger.exception("Error computing TreeExplainer.shap_values:")
            raise RuntimeError(f"SHAP computation failure: {str(e)}") from e

        # Handle multiclass output structure
        if isinstance(raw_shap, list):
            # List of 2D arrays, one per class: raw_shap[class_idx] shape is (1, n_features)
            class_shap = raw_shap[pred_class_idx][0]
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                class_shap = raw_shap[0, :, pred_class_idx]
            elif raw_shap.ndim == 2:
                class_shap = raw_shap[0, :]
            else:
                class_shap = raw_shap.flatten()
        else:
            raise ValueError(f"Unexpected SHAP values type: {type(raw_shap)}")

        # 4. Extract expected base value (prior expected value)
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = float(expected_val[pred_class_idx])
        else:
            base_value = float(expected_val)

        # 5. Build top features attribution list
        top_features: list[FeatureContribution] = []
        for i, feat_name in enumerate(expected_features):
            raw_val = float(X.iloc[0, i])
            shap_val = float(round(class_shap[i], 4))
            top_features.append(
                FeatureContribution(
                    feature=feat_name,
                    value=raw_val,
                    shap_value=shap_val
                )
            )

        # Sort feature attributions by descending absolute SHAP magnitude (|shap_value|)
        top_features.sort(key=lambda x: abs(x.shap_value), reverse=True)

        logger.info(f"SHAP explanation calculated for row {row_index}: base_value={base_value:.4f}, top_feature={top_features[0].feature} ({top_features[0].shap_value})")

        return ExplainResponse(
            prediction=predicted_label,
            confidence=confidence,
            base_value=float(round(base_value, 4)),
            top_features=top_features
        )


shap_service = SHAPService()
