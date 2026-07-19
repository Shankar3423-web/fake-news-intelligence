import logging
from fastapi import APIRouter, HTTPException, status
from app.api.schemas.retraining_schemas import RetrainingResponse
from ml.retraining.retraining_pipeline import RetrainingPipeline

router = APIRouter(prefix="/retrain", tags=["Retraining"])
logger = logging.getLogger("api.retraining")

@router.post("", response_model=RetrainingResponse, summary="Trigger model retraining")
def trigger_retraining():
    """
    Executes the automated model retraining pipeline.
    """
    logger.info("Received request to trigger model retraining.")
    
    try:
        pipeline = RetrainingPipeline()
        success = pipeline.run()
        
        # We can extract the final metadata or state from the pipeline instance if needed,
        # but the run method might just return a boolean and write to files.
        # Ideally, we should parse the retraining metadata, but returning a generic status for now.
        status_msg = "COMPLETED" if success else "FAILED"
        
        if not success:
            logger.warning("Retraining pipeline did not complete successfully.")
            # We don't raise 500 here if it's a valid ML flow failure (e.g., candidate rejected)
            # but we can communicate the result.
        else:
            logger.info("Retraining pipeline completed successfully.")
            
        return RetrainingResponse(
            status=status_msg,
            candidate_metrics={},
            comparison={},
            deployment_decision="UNKNOWN",
            reports={},
            metadata={}
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error during retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
