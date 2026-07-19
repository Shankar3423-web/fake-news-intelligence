import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from app.api.schemas.analyze_schemas import AnalyzeRequest, AnalyzeResponse
from ml.orchestrator.ultimate_pipeline import run_ultimate_pipeline
from app.database.session import get_db
from app.database.models.prediction import Prediction
from app.database.models.live_verification import LiveVerification
from app.database.models.model_version import ModelVersion
from app.database.models.user import User
from app.api.deps import get_current_user

router = APIRouter(prefix="/analyze-full", tags=["Ultimate Pipeline"])
logger = logging.getLogger("api.analyze")

@router.post("", response_model=AnalyzeResponse, summary="Run Ultimate 12-Phase Pipeline")
def analyze_news(
    request: AnalyzeRequest, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Executes the entire 12-Phase Pipeline from start to finish and saves results to PostgreSQL.
    """
    logger.info("Received request for Ultimate Pipeline Analysis")
    
    try:
        result = run_ultimate_pipeline(title=request.title, text=request.text)
        
        logger.info("Analysis complete. Saving results to database...")
        
        # 1. Look up or dynamically create ModelVersion
        pred_details = result.get("prediction_details", {})
        model_version_str = pred_details.get("model_version", "1.0.1")
        model_name = pred_details.get("model_name", "svm")
        
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
            predicted_label=result["final_verdict"],
            confidence_score=result["confidence"],
            explanation=f"Verdict decided via Consensus Engine (Agreement: {result['agreement']}).",
            model_version_id=db_model_version.id,
            user_id=current_user.id
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        # 3. Insert verification records for all matched articles
        ver_details = result.get("verification_details", {})
        matched_articles = ver_details.get("matched_articles", [])
        for article in matched_articles:
            db_verification = LiveVerification(
                prediction_id=db_prediction.id,
                fact_checking_source=article.get("source", "unknown"),
                source_url=article.get("url", "https://unknown.com"),
                verdict=ver_details.get("verification_status", "UNKNOWN"),
                verification_date=datetime.utcnow()
            )
            db.add(db_verification)
        db.commit()
        
        logger.info(f"Successfully saved prediction ID: {db_prediction.id} and {len(matched_articles)} verification records to database.")
        
        return AnalyzeResponse(
            final_verdict=result["final_verdict"],
            ml_prediction=str(result["ml_prediction"]),
            verification_status=result["verification_status"],
            agreement=result["agreement"],
            model_agreement=result["model_agreement"],
            all_model_predictions=result["all_model_predictions"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            prediction_details=result["prediction_details"],
            verification_details=result["verification_details"]
        )
    except Exception as e:
        logger.exception(f"Unexpected error in Ultimate Pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during analysis: {str(e)}"
        )
