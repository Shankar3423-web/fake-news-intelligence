from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class FeedbackRequest(BaseModel):
    prediction: str = Field(..., description="The predicted label (e.g., FAKE, REAL)")
    prediction_confidence: float = Field(..., description="Confidence score of the prediction")
    verification_status: str = Field(..., description="The verification status")
    evidence_score: float = Field(..., description="Score based on supporting evidence")
    similarity_score: float = Field(..., description="Similarity score with live news")
    final_decision: str = Field(..., description="Final decision presented to the user")
    user_feedback: str = Field(..., description="User's feedback (e.g., AGREE, DISAGREE)")
    comment: Optional[str] = Field(None, description="Optional user comment explaining their feedback")

class FeedbackResponse(BaseModel):
    feedback_id: str = Field(..., description="Unique ID of the submitted feedback")
    status: str = Field(..., description="Status of the submission")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the feedback")

class FeedbackListResponse(BaseModel):
    feedbacks: List[Dict[str, Any]] = Field(..., description="List of pending feedbacks")
