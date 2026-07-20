from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class AdminReviewRequest(BaseModel):
    feedback_id: str = Field(..., description="The unique ID of the feedback being reviewed")
    review_status: str = Field(..., description="The decision of the reviewer (e.g., APPROVED, REJECTED)")
    reviewer: str = Field(..., description="Identifier of the reviewer")
    notes: Optional[str] = Field("", description="Reviewer's notes explaining the decision")

class AdminReviewResponse(BaseModel):
    review_id: str = Field(..., description="Unique ID of the review record")
    status: str = Field(..., description="Status of the review operation")
    updated_statistics: Optional[Dict[str, Any]] = Field(None, description="Updated admin review statistics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class PendingReviewsResponse(BaseModel):
    pending_feedbacks: List[Dict[str, Any]] = Field(..., description="List of pending feedback ready for review")
