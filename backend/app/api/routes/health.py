import logging
from fastapi import APIRouter
from app.api.schemas.health_schemas import HealthResponse
import os

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger("api.health")

@router.get("", response_model=HealthResponse, summary="System health check")
def health_check():
    """
    Checks the status of the API, Model, and Dataset.
    """
    logger.info("Health check requested.")
    
    # Check model status
    model_path = os.path.join("ml", "saved_models", "best_model.pkl")
    model_status = "loaded" if os.path.exists(model_path) else "missing"
    
    # Check dataset status
    dataset_path = os.path.join("ml", "dataset", "processed", "final_features.csv")
    dataset_status = "available" if os.path.exists(dataset_path) else "missing"
    
    return HealthResponse(
        api_status="online",
        model_status=model_status,
        dataset_status=dataset_status
    )
