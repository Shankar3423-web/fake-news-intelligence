import logging
from fastapi import APIRouter, HTTPException, status
from app.api.schemas.model_schemas import ModelInfoResponse
from ml.prediction.model_loader import ModelLoader

router = APIRouter(prefix="/model", tags=["Model"])
logger = logging.getLogger("api.model")

@router.get("/info", response_model=ModelInfoResponse, summary="Get current production model info")
def get_model_info():
    """
    Retrieves information about the currently deployed production model.
    """
    logger.info("Received request for model info.")
    
    try:
        loader = ModelLoader()
        
        try:
            model, metadata, feature_order, best_model_info = loader.load_best_model()
            version = metadata.get("version", "1.0.0")
            training_date = metadata.get("training_timestamp")
            accuracy = metadata.get("validation_accuracy")
            f1_score = metadata.get("f1_score")
        except Exception as e:
            logger.warning(f"Failed to load best model metadata: {e}. Using defaults.")
            metadata = {}
            version = "1.0.0"
            training_date = None
            accuracy = None
            f1_score = None
            best_model_info = {}
        
        return ModelInfoResponse(
            current_production_model=best_model_info.get("model_key", "best_model.pkl"),
            training_date=training_date,
            accuracy=accuracy,
            f1_score=f1_score,
            version=version,
            metadata=metadata
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error retrieving model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
