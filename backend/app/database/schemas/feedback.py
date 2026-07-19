from typing import Optional
from app.database.schemas.base import BaseSchema, TimestampedSchema

class FeedbackBase(BaseSchema):
    prediction_id: int
    user_id: Optional[int] = None
    is_correct: bool
    user_comment: Optional[str] = None

class FeedbackCreate(FeedbackBase):
    pass

class FeedbackUpdate(BaseSchema):
    prediction_id: Optional[int] = None
    user_id: Optional[int] = None
    is_correct: Optional[bool] = None
    user_comment: Optional[str] = None

class FeedbackResponse(TimestampedSchema, FeedbackBase):
    pass
