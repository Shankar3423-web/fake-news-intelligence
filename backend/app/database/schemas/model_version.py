from typing import Optional, Any, Dict
from pydantic import Field
from app.database.schemas.base import BaseSchema, TimestampedSchema

class ModelVersionBase(BaseSchema):
    version_str: str = Field(..., max_length=50)
    model_name: str = Field(..., max_length=255)
    algorithm_name: str = Field(..., max_length=255)
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    filepath: str = Field(..., max_length=500)
    is_active: bool = True

class ModelVersionCreate(ModelVersionBase):
    pass

class ModelVersionUpdate(BaseSchema):
    version_str: Optional[str] = Field(None, max_length=50)
    model_name: Optional[str] = Field(None, max_length=255)
    algorithm_name: Optional[str] = Field(None, max_length=255)
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    filepath: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None

class ModelVersionResponse(TimestampedSchema, ModelVersionBase):
    pass
