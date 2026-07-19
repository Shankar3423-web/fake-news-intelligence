from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class VerificationRequest(BaseModel):
    title: Optional[str] = Field(None, description="The title of the news article", example="Breaking News: Market Hits Record High")
    text: str = Field(..., description="The full text content of the news article", example="The stock market reached an unprecedented level today as tech stocks surged...")
    prediction_response: Dict[str, Any] = Field(..., description="The response dictionary from the prediction endpoint")

class VerificationResponse(BaseModel):
    prediction: str = Field(..., description="The original predicted label (e.g., FAKE or REAL)")
    similarity_score: float = Field(..., description="Similarity score with live news sources")
    verification_status: str = Field(..., description="Verification status (e.g., VERIFIED, UNVERIFIED, CONFLICTING)")
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="List of evidence gathered from news sources")
    confidence: float = Field(..., description="Adjusted confidence score after verification")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the verification process")
