from typing import Optional
from pydantic import Field
from app.database.schemas.base import BaseSchema, TimestampedSchema

class PredictionBase(BaseSchema):
    text_content: str
    title: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=255)
    predicted_label: str = Field(..., max_length=50)
    confidence_score: float
    explanation: Optional[str] = None
    user_id: Optional[int] = None
    model_version_id: int

class PredictionCreate(PredictionBase):
    pass

class PredictionUpdate(BaseSchema):
    text_content: Optional[str] = None
    title: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=255)
    predicted_label: Optional[str] = Field(None, max_length=50)
    confidence_score: Optional[float] = None
    explanation: Optional[str] = None
    user_id: Optional[int] = None
    model_version_id: Optional[int] = None

class PredictionResponse(TimestampedSchema, PredictionBase):
    pass
