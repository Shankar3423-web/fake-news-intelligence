import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from ml.admin_review.feedback_loader import FeedbackLoader
from ml.admin_review.review_validator import ReviewValidator
from ml.admin_review.history_manager import HistoryManager
from ml.admin_review.approved_dataset_manager import ApprovedDatasetManager

logger = logging.getLogger("admin_review_pipeline")

class ReviewManager:
    """
    ReviewManager is the main coordinator for submitting, validating, and committing
    administrative decisions on collected user feedback.
    """
    def __init__(
        self,
        loader: FeedbackLoader,
        validator: ReviewValidator,
        history_mgr: HistoryManager,
        approved_mgr: ApprovedDatasetManager
    ) -> None:
        self.loader = loader
        self.validator = validator
        self.history_mgr = history_mgr
        self.approved_mgr = approved_mgr

    def submit_review(
        self,
        feedback_id: str,
        review_status: str,
        reviewer: str,
        notes: str,
        timestamp: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Submits an administrator review for a feedback record.
        Validates the transaction, records it in history, and syncs the approved dataset.
        Returns a tuple: (success: bool, error_message: str, review_record: dict)
        """
        logger.info(f"Submitting review for feedback ID '{feedback_id}' by reviewer '{reviewer}' with status '{review_status}'")

        # 1. Load prediction/verification context
        feedback_record = self.loader.load_feedback_by_id(feedback_id)
        if not feedback_record:
            err_msg = f"Feedback ID '{feedback_id}' not found in feedback history."
            logger.error(err_msg)
            return False, err_msg, None

        # 2. Extract fields from the feedback record
        prediction = feedback_record.get("Prediction")
        verification = feedback_record.get("Verification")
        decision = feedback_record.get("Decision")
        feedback_value = feedback_record.get("Feedback")

        # Use provided timestamp or current ISO 8601 time
        review_time = timestamp or datetime.now().isoformat()

        # 3. Assemble review transaction
        review_data = {
            "feedback_id": feedback_id,
            "prediction": prediction,
            "verification": verification,
            "final_decision": decision,
            "feedback_value": feedback_value,
            "review_status": review_status.upper() if isinstance(review_status, str) else review_status,
            "reviewer": reviewer,
            "timestamp": review_time
        }

        # 4. Validate transaction
        is_valid, validation_errors = self.validator.validate_review(review_data)
        if not is_valid:
            err_msg = f"Review validation failed: {'; '.join(validation_errors)}"
            logger.error(err_msg)
            return False, err_msg, None

        try:
            # 5. Record decision in history
            self.history_mgr.save_history(
                feedback_id=feedback_id,
                status=review_status,
                reviewer=reviewer,
                notes=notes,
                timestamp=review_time
            )

            # 6. Load all updated review states and sync the approved dataset
            latest_states = self.loader.get_latest_review_states(self.history_mgr.history_csv_path)
            all_feedback = self.loader.load_all_feedback()
            self.approved_mgr.sync_approved_dataset(
                latest_review_states=latest_states,
                feedback_records=all_feedback
            )

            logger.info(f"Successfully processed review decision '{review_status}' for feedback '{feedback_id}'")
            return True, None, review_data
        except Exception as e:
            err_msg = f"Database/file write error during review submission: {e}"
            logger.exception(err_msg)
            return False, err_msg, None
