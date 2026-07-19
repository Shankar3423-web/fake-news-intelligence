import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("feedback_pipeline")

class FeedbackValidator:
    """
    FeedbackValidator performs validation on the user feedback inputs,
    ensuring they conform to requirements, limits, and type annotations.
    """
    def __init__(
        self,
        allowed_values: List[str] = None,
        min_comment_len: int = 3,
        max_comment_len: int = 500,
        allow_empty_comments: bool = True
    ) -> None:
        self.allowed_values = allowed_values or ["Correct", "Incorrect", "Unsure"]
        self.min_comment_len = min_comment_len
        self.max_comment_len = max_comment_len
        self.allow_empty_comments = allow_empty_comments

    def validate_inputs(self, inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates all inputs for the feedback collection system.
        """
        errors = []
        
        # 1. Check required fields existence
        required_fields = [
            ("prediction", "Prediction"),
            ("prediction_confidence", "Prediction Confidence"),
            ("verification_status", "Verification Status"),
            ("evidence_score", "Evidence Score"),
            ("similarity_score", "Similarity Score"),
            ("final_decision", "Final Decision"),
            ("user_feedback", "Feedback")
        ]

        for field, display_name in required_fields:
            if field not in inputs:
                errors.append(f"{display_name} field is missing.")
            elif inputs[field] is None:
                errors.append(f"{display_name} is null/empty.")

        if errors:
            return False, errors

        # 2. Type & Range Validation
        # Prediction could be int or str, let's allow both
        prediction = inputs.get("prediction")
        if not isinstance(prediction, (int, str)):
            errors.append(f"Prediction must be an integer or string, got {type(prediction).__name__}.")

        # Confidence bounds [0, 1]
        confidence = inputs.get("prediction_confidence")
        if not isinstance(confidence, (int, float)):
            errors.append(f"Prediction Confidence must be numeric, got {type(confidence).__name__}.")
        elif not (0.0 <= float(confidence) <= 1.0):
            errors.append(f"Prediction Confidence must be in range [0, 1], got {confidence}.")

        # Verification Status must be string
        verification_status = inputs.get("verification_status")
        if not isinstance(verification_status, str):
            errors.append(f"Verification Status must be a string, got {type(verification_status).__name__}.")

        # Evidence score bounds [0, 1]
        evidence_score = inputs.get("evidence_score")
        if not isinstance(evidence_score, (int, float)):
            errors.append(f"Evidence Score must be numeric, got {type(evidence_score).__name__}.")
        elif not (0.0 <= float(evidence_score) <= 1.0):
            errors.append(f"Evidence Score must be in range [0, 1], got {evidence_score}.")

        # Similarity score bounds [0, 1]
        similarity_score = inputs.get("similarity_score")
        if not isinstance(similarity_score, (int, float)):
            errors.append(f"Similarity Score must be numeric, got {type(similarity_score).__name__}.")
        elif not (0.0 <= float(similarity_score) <= 1.0):
            errors.append(f"Similarity Score must be in range [0, 1], got {similarity_score}.")

        # Final Decision must be string
        final_decision = inputs.get("final_decision")
        if not isinstance(final_decision, str):
            errors.append(f"Final Decision must be a string, got {type(final_decision).__name__}.")

        # 3. User Feedback Value validation
        user_feedback = inputs.get("user_feedback")
        if not isinstance(user_feedback, str):
            errors.append(f"User Feedback must be a string, got {type(user_feedback).__name__}.")
        elif user_feedback not in self.allowed_values:
            errors.append(f"Invalid Feedback Value: '{user_feedback}'. Allowed values: {self.allowed_values}.")

        # 4. Optional Comment Length validation
        comment = inputs.get("comment")
        if comment is not None:
            if not isinstance(comment, str):
                errors.append(f"Optional comment must be a string, got {type(comment).__name__}.")
            else:
                comment_len = len(comment)
                if comment_len == 0:
                    if not self.allow_empty_comments:
                        errors.append("Comment cannot be empty.")
                else:
                    if comment_len < self.min_comment_len or comment_len > self.max_comment_len:
                        errors.append(f"Comment length {comment_len} is out of bounds [{self.min_comment_len}, {self.max_comment_len}].")

        # 5. Optional Timestamp format validation
        timestamp = inputs.get("timestamp")
        if timestamp is not None:
            if not isinstance(timestamp, str):
                errors.append(f"Timestamp must be a string, got {type(timestamp).__name__}.")
            else:
                try:
                    # Validate ISO format
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    errors.append(f"Timestamp '{timestamp}' is not in valid ISO 8601 format.")

        return len(errors) == 0, errors
