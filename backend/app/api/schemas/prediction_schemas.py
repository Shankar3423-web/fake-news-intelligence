from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class PredictionRequest(BaseModel):
    title: str = Field(..., description="The title of the news article", example="Breaking News: Market Hits Record High")
    text: str = Field(..., description="The full text content of the news article", example="The stock market reached an unprecedented level today as tech stocks surged...")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="The predicted label (e.g., FAKE or REAL)")
    confidence: float = Field(..., description="Confidence score of the prediction (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities for FAKE and REAL")
    model_version: str = Field(..., description="The version of the model used for prediction")
    processing_time: float = Field(..., description="Time taken to process the prediction in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the prediction")
