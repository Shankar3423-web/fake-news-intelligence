from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class AnalyzeRequest(BaseModel):
    title: Optional[str] = Field(None, description="Title of the news article")
    text: str = Field(..., description="Full text of the news article to be analyzed")

class ModelPrediction(BaseModel):
    model_name: str = Field(..., description="Name of the model (e.g., svm, logistic_regression)")
    algorithm: str = Field(..., description="Algorithm used (e.g., Linear SVM, Logistic Regression)")
    prediction: str = Field(..., description="Model's prediction: REAL or FAKE")
    confidence: float = Field(..., description="Model's confidence score (0.0 to 1.0)")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")

class AnalyzeResponse(BaseModel):
    final_verdict: str = Field(..., description="The definitive prediction: REAL or FAKE")
    ml_prediction: str = Field(..., description="ML majority vote prediction: REAL or FAKE")
    verification_status: str = Field(..., description="Live API verification result: VERIFIED_REAL, VERIFIED_FAKE, or INCONCLUSIVE")
    agreement: str = Field(..., description="Whether ML and APIs agreed: FULL_AGREEMENT, API_OVERRIDE, or ML_FALLBACK")
    model_agreement: str = Field(..., description="Multi-model vote summary (e.g., '3/4 models agree: REAL')")
    all_model_predictions: List[ModelPrediction] = Field(..., description="Individual predictions from all 4 ML models")
    confidence: float = Field(..., description="Overall confidence score (0.0 to 1.0)")
    processing_time: float = Field(..., description="Total time taken across all phases in seconds")
    prediction_details: Dict[str, Any] = Field(..., description="Full output from the best ML model (Phase 8)")
    verification_details: Dict[str, Any] = Field(..., description="Full output from the Live News Verification Engine (Phase 9)")
