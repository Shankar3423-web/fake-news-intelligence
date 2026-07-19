from typing import Optional
from app.database.schemas.base import BaseSchema, TimestampedSchema

class ApprovedDatasetBase(BaseSchema):
    dataset_id: int
    approved_by: int
    approval_notes: Optional[str] = None

class ApprovedDatasetCreate(ApprovedDatasetBase):
    pass

class ApprovedDatasetUpdate(BaseSchema):
    dataset_id: Optional[int] = None
    approved_by: Optional[int] = None
    approval_notes: Optional[str] = None

class ApprovedDatasetResponse(TimestampedSchema, ApprovedDatasetBase):
    pass
