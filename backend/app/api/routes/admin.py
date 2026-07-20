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
        
        try:
            fb_id_int = int(request.feedback_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid feedback_id format. Must be an integer ID.")

        feedback = db.query(Feedback).filter(Feedback.id == fb_id_int).first()
        if not feedback:
            raise HTTPException(status_code=404, detail=f"Feedback ID {request.feedback_id} not found in database.")
            
        status_upper = request.review_status.upper().strip()
        if status_upper in ("ACCEPT", "ACCEPTED", "APPROVED"):
            status_upper = "APPROVED"
        elif status_upper in ("REJECT", "REJECTED", "DECLINED"):
            status_upper = "REJECTED"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid review_status: '{request.review_status}'. Allowed values are APPROVED/ACCEPT or REJECTED/REJECT.")

        feedback.status = status_upper
        db.commit()
        db.refresh(feedback)
        
        # Try to sync with local ML pipeline files if available
        try:
            from ml.admin_review.admin_review_pipeline import run_admin_review_pipeline
            run_admin_review_pipeline(
                feedback_id=str(feedback.id),
                review_status=status_upper,
                reviewer=request.reviewer or "Admin",
                notes=request.notes or ""
            )
        except Exception as ml_err:
            logger.warning(f"ML pipeline sync skipped/failed during admin review: {ml_err}")

        return AdminReviewResponse(
            review_id=str(feedback.id),
            status="SUCCESS",
            updated_statistics={"total_reviewed": 1, "final_status": status_upper},
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
            headline = f.prediction.title if (f.prediction and f.prediction.title) else (
                f.prediction.text_content[:60] + "..." if (f.prediction and f.prediction.text_content) else "News Article"
            )
            pred_label = (f.prediction.predicted_label if f.prediction else "REAL").upper()
            
            # Calculate explicit User Correction (what label the user asserts the news actually is)
            if f.is_correct:
                user_correction = pred_label
                user_feedback = "AGREE"
            else:
                user_correction = "FAKE" if pred_label == "REAL" else "REAL"
                user_feedback = "DISAGREE"

            pending_records.append({
                # Standard Title Case Keys
                "Feedback ID": str(f.id),
                "Prediction ID": str(f.prediction_id) if f.prediction_id else "N/A",
                "Timestamp": f.created_at.isoformat() if f.created_at else None,
                "Prediction": pred_label,
                "User Correction": user_correction,
                "User Feedback": user_feedback,
                "Is Correct": f.is_correct,
                "Comment": f.user_comment or "No comment provided",
                "User Comments": f.user_comment or "No comment provided",
                "Headline": headline,
                "News Title": headline,
                "Article Title": headline,

                # Compatibility Keys (snake_case & lowercase)
                "feedback_id": str(f.id),
                "prediction_id": str(f.prediction_id) if f.prediction_id else "N/A",
                "timestamp": f.created_at.isoformat() if f.created_at else None,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "prediction": pred_label,
                "predicted_label": pred_label,
                "model_prediction": pred_label,
                "user_correction": user_correction,
                "user_feedback": user_feedback,
                "user_comment": f.user_comment or "No comment provided",
                "user_comments": f.user_comment or "No comment provided",
                "is_correct": f.is_correct,
                "headline": headline,
                "title": headline,
                "news_title": headline,
                "article_title": headline,

                # Additional Meta
                "Verification": f.prediction.live_verifications[0].verdict if (f.prediction and f.prediction.live_verifications) else "UNVERIFIED",
                "Final Decision": pred_label,
                "Similarity Score": 0.0,
                "Evidence Score": 0.0,
                "Confidence": f.prediction.confidence_score if f.prediction else 0.0,
            })
            
        return PendingReviewsResponse(pending_feedbacks=pending_records)
    except Exception as e:
        logger.exception(f"Unexpected error loading pending reviews: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

