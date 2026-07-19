import time
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.api.schemas.verification_schemas import VerificationRequest, VerificationResponse
from ml.verification.verification_pipeline import run_verification_pipeline
from app.database.session import get_db
from app.database.models.prediction import Prediction
from app.database.models.live_verification import LiveVerification

router = APIRouter(prefix="/verify-news", tags=["Verification"])
logger = logging.getLogger("api.verification")

@router.post("", response_model=VerificationResponse, summary="Verify news against live sources")
def verify_news(request: VerificationRequest, db: Session = Depends(get_db)):
    """
    Executes the live news verification pipeline, gathers evidence, and saves results to the database.
    """
    logger.info(f"Received verification request for title: {request.title}")
    
    try:
        # Verification pipeline expects article_text, prediction_response, and article_title
        result = run_verification_pipeline(
            article_text=request.text,
            prediction_response=request.prediction_response,
            article_title=request.title
        )
        
        if result.get("status") == "error":
            logger.error(f"Verification failed: {result.get('message')}")
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown verification error"))
            
        logger.info("Verification successful. Saving results to database...")
        
        # 1. Find matching prediction by text content or fallback to the most recent prediction
        db_prediction = db.query(Prediction).filter(Prediction.text_content == request.text).order_by(Prediction.id.desc()).first()
        if not db_prediction:
            db_prediction = db.query(Prediction).order_by(Prediction.id.desc()).first()
            
        # 2. Insert verification records linked to the prediction (if one exists)
        if db_prediction:
            matched_articles = result.get("matched_articles", [])
            for article in matched_articles:
                db_verification = LiveVerification(
                    prediction_id=db_prediction.id,
                    fact_checking_source=article.get("source", "unknown"),
                    source_url=article.get("url", "https://unknown.com"),
                    verdict=result.get("verification_status", "UNKNOWN"),
                    verification_date=datetime.utcnow()
                )
                db.add(db_verification)
            db.commit()
            logger.info(f"Saved {len(matched_articles)} verification records to database linked to Prediction ID {db_prediction.id}")
        else:
            logger.warning("No prediction record found to link this verification to. Verification results were not written to database.")
            
        return VerificationResponse(
            prediction=result.get("original_prediction", "UNKNOWN"),
            similarity_score=result.get("similarity_score", 0.0),
            verification_status=result.get("verification_status", "UNKNOWN"),
            supporting_evidence=result.get("supporting_evidence", []),
            confidence=result.get("adjusted_confidence", 0.0),
            metadata=result.get("metadata")
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error during verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
