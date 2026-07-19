import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger("admin_review_pipeline")

class FeedbackLoader:
    """
    FeedbackLoader loads collected user feedback from Phase 10 feedback history
    and identifies pending records based on administrative review history.
    """
    def __init__(self, feedback_history_path: str) -> None:
        self.feedback_history_path = feedback_history_path

    def load_all_feedback(self) -> List[Dict[str, Any]]:
        """
        Loads all feedback records from the feedback history CSV.
        """
        if not os.path.exists(self.feedback_history_path):
            logger.warning(f"Feedback history file not found at {self.feedback_history_path}")
            return []

        try:
            df = pd.read_csv(self.feedback_history_path)
            # Standardize column headers and clean up empty cells
            df = df.where(pd.notnull(df), None)
            records = df.to_dict(orient="records")
            logger.debug(f"Loaded {len(records)} feedback records from {self.feedback_history_path}")
            return records
        except Exception as e:
            logger.error(f"Error loading feedback history from {self.feedback_history_path}: {e}")
            return []

    def get_latest_review_states(self, review_history_path: str) -> Dict[str, str]:
        """
        Parses the review history and returns a mapping of Feedback ID -> Latest Review Status.
        """
        if not os.path.exists(review_history_path):
            logger.debug(f"Review history file not found at {review_history_path}. No records reviewed yet.")
            return {}

        try:
            df = pd.read_csv(review_history_path)
            if df.empty:
                return {}
            
            # Sort by index or timestamp to get the latest review decision
            # For simplicity, since we append, we can iterate forward and overwrite
            latest_status = {}
            for _, row in df.iterrows():
                fid = str(row.get("Feedback ID", "")).strip()
                status = str(row.get("Review Status", "")).strip().upper()
                if fid and status:
                    latest_status[fid] = status
            return latest_status
        except Exception as e:
            logger.error(f"Error loading review history from {review_history_path}: {e}")
            return {}

    def load_pending_feedback(self, review_history_path: str) -> List[Dict[str, Any]]:
        """
        Loads the feedback records that are pending review.
        A record is pending if:
        1. It has never been reviewed (not in review history).
        2. Its latest review decision in the review history is PENDING.
        """
        all_feedback = self.load_all_feedback()
        if not all_feedback:
            return []

        review_states = self.get_latest_review_states(review_history_path)
        pending_records = []

        for record in all_feedback:
            fid = str(record.get("Feedback ID", "")).strip()
            if not fid:
                continue

            status = review_states.get(fid)
            if status is None or status == "PENDING":
                pending_records.append(record)

        logger.info(f"Identified {len(pending_records)} pending feedback records out of {len(all_feedback)} total.")
        return pending_records

    def load_feedback_by_id(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads a single feedback record by its Feedback ID.
        """
        all_feedback = self.load_all_feedback()
        for record in all_feedback:
            if str(record.get("Feedback ID", "")).strip() == feedback_id:
                return record
        logger.warning(f"Feedback record with ID {feedback_id} not found.")
        return None
