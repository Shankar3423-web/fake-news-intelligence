import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("admin_review_pipeline")

class ReviewValidator:
    """
    ReviewValidator performs validation on the administrator review transactions,
    ensuring all feedback and review attributes conform to schema requirements.
    """
    def __init__(self, allowed_states: List[str] = None) -> None:
        self.allowed_states = allowed_states or ["APPROVED", "REJECTED", "PENDING"]
        self.allowed_feedback_values = ["Correct", "Incorrect", "Unsure", "AGREE", "DISAGREE"]

    def validate_review(self, review_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates all fields in the administrator review transaction.
        Expects a dictionary with keys:
          - feedback_id: str
          - prediction: int/str
          - verification: str
          - final_decision: str
          - feedback_value: str
          - review_status: str
          - reviewer: str
          - timestamp: str
        """
        errors = []

        # 1. Required Fields Existence Check
        required_fields = [
            ("feedback_id", "Feedback ID"),
            ("prediction", "Prediction"),
            ("verification", "Verification"),
            ("final_decision", "Final Decision"),
            ("feedback_value", "Feedback Value"),
            ("review_status", "Review Status"),
            ("reviewer", "Reviewer Name"),
            ("timestamp", "Timestamp")
        ]

        for field, display_name in required_fields:
            if field not in review_data:
                errors.append(f"Required review field '{field}' ({display_name}) is missing.")
            elif review_data[field] is None:
                errors.append(f"Field '{field}' ({display_name}) cannot be null or empty.")

        if errors:
            return False, errors

        # 2. Type and Format Validations

        # Feedback ID
        feedback_id = review_data.get("feedback_id")
        if not isinstance(feedback_id, str) or not feedback_id.strip():
            errors.append("Feedback ID must be a non-empty string.")

        # Prediction
        prediction = review_data.get("prediction")
        if not isinstance(prediction, (int, str)):
            errors.append(f"Prediction must be an integer or string, got {type(prediction).__name__}.")

        # Verification Status
        verification = review_data.get("verification")
        if not isinstance(verification, str) or not verification.strip():
            errors.append("Verification must be a non-empty string.")

        # Final Decision
        final_decision = review_data.get("final_decision")
        if not isinstance(final_decision, str) or not final_decision.strip():
            errors.append("Final Decision must be a non-empty string.")

        # Feedback Value
        feedback_value = review_data.get("feedback_value")
        if not isinstance(feedback_value, str):
            errors.append("Feedback Value must be a string.")
        elif feedback_value not in self.allowed_feedback_values:
            errors.append(f"Invalid Feedback Value '{feedback_value}'. Allowed values: {self.allowed_feedback_values}.")

        # Review Status
        review_status = review_data.get("review_status")
        if not isinstance(review_status, str):
            errors.append("Review Status must be a string.")
        elif review_status.upper() not in self.allowed_states:
            errors.append(f"Invalid Review Status '{review_status}'. Allowed values: {self.allowed_states}.")

        # Reviewer Name
        reviewer = review_data.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            errors.append("Reviewer Name must be a non-empty string.")

        # Timestamp
        timestamp = review_data.get("timestamp")
        if not isinstance(timestamp, str):
            errors.append("Timestamp must be a string.")
        else:
            try:
                # Validate ISO format
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                errors.append(f"Timestamp '{timestamp}' is not in valid ISO 8601 format.")

        return len(errors) == 0, errors
