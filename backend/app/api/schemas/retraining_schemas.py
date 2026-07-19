from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class RetrainingResponse(BaseModel):
    status: str = Field(..., description="Status of the retraining process (e.g., COMPLETED, FAILED, SKIPPED)")
    candidate_metrics: Optional[Dict[str, Any]] = Field(None, description="Metrics of the newly trained candidate model")
    comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison between the new candidate and the current production model")
    deployment_decision: Optional[str] = Field(None, description="Decision on whether the candidate model was deployed (e.g., DEPLOYED, REJECTED)")
    reports: Optional[Dict[str, Any]] = Field(None, description="Paths to generated reports")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
