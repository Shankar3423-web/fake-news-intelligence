from typing import Optional
from pydantic import Field
from app.database.schemas.base import BaseSchema, TimestampedSchema

class VerificationQueueBase(BaseSchema):
    text_content: str
    title: Optional[str] = Field(None, max_length=500)
    source_url: Optional[str] = Field(None, max_length=1000)
    status: str = Field("PENDING", max_length=50)
    assigned_to: Optional[int] = None

class VerificationQueueCreate(VerificationQueueBase):
    pass

class VerificationQueueUpdate(BaseSchema):
    text_content: Optional[str] = None
    title: Optional[str] = Field(None, max_length=500)
    source_url: Optional[str] = Field(None, max_length=1000)
    status: Optional[str] = Field(None, max_length=50)
    assigned_to: Optional[int] = None

class VerificationQueueResponse(TimestampedSchema, VerificationQueueBase):
    pass
