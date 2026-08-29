"""Model inference API endpoint for batch CSV file classification."""

import io
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
import pandas as pd

from app.services.model_loader import model_loader
from app.schemas.predict import PredictionResponse
from app.services.predictor import (
    PredictionService,
    EmptyDataError,
    MissingFeaturesError,
    PredictionError,
)

router = APIRouter()
logger = logging.getLogger("PredictAPI")


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Multiclass Intrusion Prediction",
    description="Upload a CSV file containing network flow records to classify benign traffic vs security attacks.",
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classifies uploaded CSV network flow records using the Top-14 XGBoost model."""
    logger.info(f"Received prediction request for file: '{file.filename}' (content_type: {file.content_type})")

    # 1. Validate file extension
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension for '{filename}'. Only CSV files (.csv) are supported."
        )

    # 2. Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file contents."
        ) from e

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    # 3. Parse CSV content into pandas DataFrame (filtered to target feature set for high throughput)
    expected_set = set(model_loader.expected_features)
    try:
        df = pd.read_csv(io.BytesIO(content), usecols=lambda col: col in expected_set, engine="c")
    except pd.errors.EmptyDataError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded CSV file contains no data or valid headers."
        ) from e
    except Exception as e:
        logger.warning(f"Failed to parse CSV file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid CSV format: {str(e)}"
        ) from e

    # 4. Execute prediction service
    try:
        response = PredictionService.predict_dataframe(df)
        return response
    except EmptyDataError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except MissingFeaturesError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Missing Required Features",
                "message": str(e),
                "missing_features": e.missing_features,
            }
        ) from e
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference execution error: {str(e)}"
        ) from e
    except Exception as e:
        logger.exception("Unexpected error during prediction processing:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred during prediction inference."
        ) from e
