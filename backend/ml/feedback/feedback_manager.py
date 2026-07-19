import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("feedback_pipeline")

class FeedbackManager:
    """
    FeedbackManager coordinates feedback record generation and compilation.
    """
    def __init__(self, system_version: str = "1.0.0") -> None:
        self.system_version = system_version

    def generate_feedback_id(self) -> str:
        """Generates a unique feedback record identifier."""
        return f"fb_{uuid.uuid4().hex}"

    def create_record(
        self,
        prediction: Any,
        prediction_confidence: float,
        verification_status: str,
        evidence_score: float,
        similarity_score: float,
        final_decision: str,
        user_feedback: str,
        comment: str,
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compiles the complete feedback record dictionary.
        """
        fb_id = self.generate_feedback_id()
        record_time = timestamp if timestamp else datetime.now().isoformat()
        
        record = {
            "feedback_id": fb_id,
            "timestamp": record_time,
            "prediction": prediction,
            "prediction_confidence": float(prediction_confidence),
            "verification_status": verification_status,
            "evidence_score": float(evidence_score),
            "similarity_score": float(similarity_score),
            "final_decision": final_decision,
            "feedback_value": user_feedback,
            "comment": comment,
            "system_version": self.system_version
        }
        
        logger.debug(f"Generated feedback record: {fb_id}")
        return record
