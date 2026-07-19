import time
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.api.schemas.feedback_schemas import FeedbackRequest, FeedbackResponse, FeedbackListResponse
from ml.feedback.feedback_pipeline import run_feedback_pipeline
from app.database.session import get_db
from app.database.models.prediction import Prediction
from app.database.models.feedback import Feedback
from app.database.models.user import User
from app.api.deps import get_current_user

router = APIRouter(prefix="/feedback", tags=["Feedback"])
logger = logging.getLogger("api.feedback")

@router.post("", response_model=FeedbackResponse, summary="Submit user feedback")
def submit_feedback(
    request: FeedbackRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submits user feedback for a prediction and verification result and saves it to the database.
    """
    logger.info("Received feedback submission")
    
    try:
        result = run_feedback_pipeline(
            prediction=request.prediction,
            prediction_confidence=request.prediction_confidence,
            verification_status=request.verification_status,
            evidence_score=request.evidence_score,
            similarity_score=request.similarity_score,
            final_decision=request.final_decision,
            user_feedback=request.user_feedback,
            comment=request.comment
        )
        
        if result.get("status") in ("error", "FAILED"):
            error_msg = result.get("error") or result.get("message") or "Unknown feedback processing error"
            logger.error(f"Feedback processing failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
            
        logger.info("Feedback successfully processed by ML pipeline. Saving to database...")
        
        # 1. Find the last prediction in the database for THIS user to link this feedback to
        db_prediction = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.id.desc()).first()
        
        # 2. Insert feedback record
        if db_prediction:
            is_correct_val = request.user_feedback.upper() in ("AGREE", "CORRECT", "TRUE")
            db_feedback = Feedback(
                prediction_id=db_prediction.id,
                user_id=current_user.id,
                is_correct=is_correct_val,
                user_comment=request.comment
            )
            db.add(db_feedback)
            db.commit()
            db.refresh(db_feedback)
            logger.info(f"Feedback successfully saved to database with ID: {db_feedback.id}")
        else:
            logger.warning("No prediction record found in database. Feedback saved by pipeline but skipped in database.")
            
        return FeedbackResponse(
            feedback_id=result.get("feedback_id", "unknown"),
            status="SUCCESS",
            metadata=result.get("metadata")
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error processing feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get("", response_model=FeedbackListResponse, summary="Get feedback list")
def get_feedback_list(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves all user feedback from the database for the current user.
    """
    try:
        from sqlalchemy import desc
        feedbacks = db.query(Feedback).filter(Feedback.user_id == current_user.id).order_by(desc(Feedback.created_at)).all()
        
        feedback_list = []
        for f in feedbacks:
            feedback_list.append({
                "feedback_id": str(f.id),
                "prediction_id": f.prediction_id,
                "is_correct": f.is_correct,
                "user_comment": f.user_comment,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "prediction_title": f.prediction.title if f.prediction else "Unknown Article",
                "prediction_label": f.prediction.predicted_label if f.prediction else "UNKNOWN",
                "prediction_confidence": f.prediction.confidence_score if f.prediction else 0.0
            })
        return FeedbackListResponse(feedbacks=feedback_list)
    except Exception as e:
        logger.exception(f"Error fetching feedbacks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching feedbacks: {str(e)}"
        )
