from datetime import datetime
from typing import Optional
from pydantic import Field
from app.database.schemas.base import BaseSchema, TimestampedSchema

class LiveVerificationBase(BaseSchema):
    prediction_id: int
    fact_checking_source: str = Field(..., max_length=255)
    source_url: str = Field(..., max_length=1000)
    verdict: str = Field(..., max_length=100)
    verification_date: datetime

class LiveVerificationCreate(LiveVerificationBase):
    pass

class LiveVerificationUpdate(BaseSchema):
    prediction_id: Optional[int] = None
    fact_checking_source: Optional[str] = Field(None, max_length=255)
    source_url: Optional[str] = Field(None, max_length=1000)
    verdict: Optional[str] = Field(None, max_length=100)
    verification_date: Optional[datetime] = None

class LiveVerificationResponse(TimestampedSchema, LiveVerificationBase):
    pass
