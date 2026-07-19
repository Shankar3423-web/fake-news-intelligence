from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ModelInfoResponse(BaseModel):
    current_production_model: str = Field(..., description="Name or path of the current production model")
    training_date: Optional[str] = Field(None, description="Timestamp of when the model was trained")
    accuracy: Optional[float] = Field(None, description="Accuracy of the model")
    f1_score: Optional[float] = Field(None, description="F1 score of the model")
    version: str = Field(..., description="Version of the model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the model")
