import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.api.schemas.admin_schemas import AdminReviewRequest, AdminReviewResponse, PendingReviewsResponse
from app.database.session import get_db

router = APIRouter(prefix="/admin", tags=["Admin Review"])
logger = logging.getLogger("api.admin")

@router.post("/review", response_model=AdminReviewResponse, summary="Review pending feedback")
def review_feedback(
    request: AdminReviewRequest,
    db: Session = Depends(get_db)
):
    """
    Processes an administrative review for pending user feedback.
    """
    logger.info(f"Received admin review for feedback_id: {request.feedback_id}")
    
    try:
        from app.database.models.feedback import Feedback
        
        feedback = db.query(Feedback).filter(Feedback.id == int(request.feedback_id)).first()
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
            
        feedback.status = request.review_status.upper()
        db.commit()
        db.refresh(feedback)
        
        return AdminReviewResponse(
            review_id=str(feedback.id),
            status="SUCCESS",
            updated_statistics={"total_reviewed": 1},
            metadata={"reviewer": request.reviewer, "notes": request.notes}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error processing admin review: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get("/pending", response_model=PendingReviewsResponse, summary="Get pending feedback for review")
def get_pending_reviews(
    db: Session = Depends(get_db)
):
    """
    Retrieves the list of feedback awaiting admin review.
    """
    try:
        from app.database.models.feedback import Feedback
        
        pending_feedbacks = db.query(Feedback).filter(Feedback.status == "PENDING").all()
        
        pending_records = []
        for f in pending_feedbacks:
            pending_records.append({
                "Feedback ID": str(f.id),
                "Timestamp": f.created_at.isoformat() if f.created_at else None,
                "Prediction": f.prediction.predicted_label if f.prediction else "UNKNOWN",
                "Verification": f.prediction.live_verifications[0].verdict if (f.prediction and f.prediction.live_verifications) else "UNVERIFIED",
                "Final Decision": f.prediction.predicted_label if f.prediction else "UNKNOWN",
                "User Feedback": "AGREE" if f.is_correct else "DISAGREE",
                "Comment": f.user_comment,
                "Similarity Score": 0.0,
                "Evidence Score": 0.0,
                "Confidence": f.prediction.confidence_score if f.prediction else 0.0
            })
            
        return PendingReviewsResponse(pending_feedbacks=pending_records)
    except Exception as e:
        logger.exception(f"Unexpected error loading pending reviews: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
