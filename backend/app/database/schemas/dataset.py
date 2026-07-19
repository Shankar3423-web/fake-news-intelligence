from typing import Optional
from pydantic import Field
from app.database.schemas.base import BaseSchema, TimestampedSchema

class DatasetBase(BaseSchema):
    name: str = Field(..., max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    source_path: str = Field(..., max_length=500)
    num_rows: Optional[int] = None

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(BaseSchema):
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    source_path: Optional[str] = Field(None, max_length=500)
    num_rows: Optional[int] = None

class DatasetResponse(TimestampedSchema, DatasetBase):
    pass
