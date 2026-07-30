"""SHAP explainability API endpoint for local feature attribution analysis."""

import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.explain import ExplainRequest, ExplainResponse
from app.services.shap_service import shap_service

router = APIRouter()
logger = logging.getLogger("ExplainAPI")


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Local SHAP Feature Attribution Explanation",
    description="Compute TreeExplainer SHAP values and base value for a single network flow instance.",
)
async def explain(request: ExplainRequest) -> ExplainResponse:
    """Computes SHAP feature attributions for a single network flow prediction instance."""
    logger.info(f"Received SHAP explanation request for row index {request.row}")

    if not request.features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Features dictionary cannot be empty."
        )

    try:
        response = shap_service.explain_instance(request.features, row_index=request.row)
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"SHAP explanation calculation failed: {str(e)}"
        ) from e
    except Exception as e:
        logger.exception("Unexpected error during SHAP explanation computation:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred while computing SHAP attributions."
        ) from e
