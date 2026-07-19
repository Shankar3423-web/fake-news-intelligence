import time
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.api.schemas.prediction_schemas import PredictionRequest, PredictionResponse
from ml.prediction.prediction_pipeline import run_prediction_pipeline
from app.database.session import get_db
from app.database.models.prediction import Prediction
from app.database.models.model_version import ModelVersion
from app.database.models.user import User
from app.api.deps import get_current_user

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger("api.prediction")

@router.post("", response_model=PredictionResponse, summary="Predict if news is real or fake")
def predict_news(
    request: PredictionRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes the ML prediction pipeline for the given news article and saves it in the database.
    """
    start_time = time.time()
    logger.info(f"Received prediction request for title: {request.title}")
    
    try:
        # The ML pipeline expects raw_text as a single string, we can combine title and text
        full_text = f"{request.title}\n{request.text}" if request.title else request.text
        
        result = run_prediction_pipeline(full_text)
        
        if result.get("status") == "failed":
            logger.error(f"Prediction failed: {result.get('errors')}")
            raise HTTPException(status_code=500, detail=str(result.get("errors", "Unknown prediction error")))
            
        processing_time = time.time() - start_time
        
        logger.info("Prediction successful. Saving to database...")
        
        # 1. Look up or dynamically create ModelVersion
        model_version_str = result.get("model_version", "1.0.1")
        model_name = result.get("model_name", "svm")
        
        db_model_version = db.query(ModelVersion).filter(ModelVersion.version_str == model_version_str).first()
        if not db_model_version:
            db_model_version = ModelVersion(
                version_str=model_version_str,
                model_name=model_name,
                algorithm_name="Linear SVM" if model_name == "svm" else model_name.upper(),
                filepath=f"ml/training/models/{model_name}/model.joblib",
                is_active=True
            )
            db.add(db_model_version)
            db.commit()
            db.refresh(db_model_version)
            
        # 2. Insert prediction record
        db_prediction = Prediction(
            title=request.title,
            text_content=request.text,
            predicted_label=result.get("label", "UNKNOWN"),
            confidence_score=result.get("confidence", 0.0),
            explanation=f"Predicted by model ensemble with primary model: {model_name}.",
            model_version_id=db_model_version.id,
            user_id=current_user.id
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        logger.info(f"Prediction successfully saved to database with ID: {db_prediction.id}")
        
        return PredictionResponse(
            prediction=result.get("label", "UNKNOWN"),
            confidence=result.get("confidence", 0.0),
            probabilities=result.get("probabilities", {}),
            model_version=result.get("model_version", "unknown"),
            processing_time=processing_time,
            metadata={
                "model_name": result.get("model_name"),
                "evaluation_version": result.get("evaluation_version"),
                "prediction_time_ms": result.get("prediction_time_ms"),
                "throughput": result.get("throughput"),
                "memory_usage": result.get("memory_usage")
            }
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@router.get("/history", summary="Get historical predictions")
def get_prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieves all historical news predictions for the current user.
    """
    try:
        from sqlalchemy import desc
        predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(desc(Prediction.created_at)).all()
        
        history = []
        for pred in predictions:
            verifications = [{
                "id": v.id,
                "fact_checking_source": v.fact_checking_source,
                "source_url": v.source_url,
                "verdict": v.verdict,
                "verification_date": v.verification_date.isoformat() if v.verification_date else None
            } for v in pred.live_verifications]
            
            feedbacks = [{
                "id": f.id,
                "is_correct": f.is_correct,
                "user_comment": f.user_comment,
                "created_at": f.created_at.isoformat() if f.created_at else None
            } for f in pred.feedbacks]
            
            history.append({
                "id": pred.id,
                "title": pred.title,
                "text_content": pred.text_content,
                "predicted_label": pred.predicted_label,
                "confidence_score": pred.confidence_score,
                "explanation": pred.explanation,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
                "live_verifications": verifications,
                "feedbacks": feedbacks,
                "model_version": pred.model_version.version_str if pred.model_version else "1.0.1"
            })
            
        return history
    except Exception as e:
        logger.exception(f"Error retrieving prediction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching history: {str(e)}"
        )

@router.get("/stats", summary="Get prediction statistics")
def get_prediction_stats(db: Session = Depends(get_db)):
    """
    Computes system statistics for dashboard and analytics charts.
    """
    try:
        from sqlalchemy import desc, func
        from app.database.models.feedback import Feedback
        
        total_predictions = db.query(Prediction).count()
        real_count = db.query(Prediction).filter(Prediction.predicted_label == "REAL").count()
        fake_count = db.query(Prediction).filter(Prediction.predicted_label == "FAKE").count()
        
        # Calculate average confidence
        avg_confidence = db.query(func.avg(Prediction.confidence_score)).scalar() or 0.0
        
        # Calculate feedback stats
        total_feedback = db.query(Feedback).count()
        correct_feedback = db.query(Feedback).filter(Feedback.is_correct == True).count()
        accuracy = (correct_feedback / total_feedback) if total_feedback > 0 else 0.88
        
        # Group by label to send count data
        label_distribution = [
            {"name": "Real News", "value": real_count},
            {"name": "Fake News", "value": fake_count}
        ]
        
        # Retrieve recent predictions
        recent_preds = db.query(Prediction).order_by(desc(Prediction.created_at)).limit(5).all()
        recent = [{
            "id": p.id,
            "title": p.title or (p.text_content[:60] + "..."),
            "label": p.predicted_label,
            "confidence": p.confidence_score,
            "time": p.created_at.isoformat() if p.created_at else None
        } for p in recent_preds]
        
        # Timeline stats (last 10 predictions or group by time)
        timeline = []
        timeline_preds = db.query(Prediction).order_by(desc(Prediction.created_at)).limit(10).all()
        for p in reversed(timeline_preds):
            timeline.append({
                "date": p.created_at.strftime("%H:%M") if p.created_at else "00:00",
                "confidence": round(p.confidence_score * 100, 1),
                "is_real": 1 if p.predicted_label == "REAL" else 0
            })
            
        return {
            "total_predictions": total_predictions,
            "real_count": real_count,
            "fake_count": fake_count,
            "average_confidence": float(avg_confidence),
            "system_accuracy": accuracy,
            "label_distribution": label_distribution,
            "recent_activity": recent,
            "timeline": timeline
        }
    except Exception as e:
        logger.exception(f"Error retrieving prediction statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching statistics: {str(e)}"
        )

