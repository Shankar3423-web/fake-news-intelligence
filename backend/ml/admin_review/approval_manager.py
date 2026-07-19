import logging
from typing import Dict, Any, List

logger = logging.getLogger("admin_review_pipeline")

class ApprovalManager:
    """
    ApprovalManager coordinates the validation of approved feedback records
    and determines eligibility for model retraining.
    """
    def __init__(self, min_retraining_records: int = 5) -> None:
        self.min_retraining_records = min_retraining_records

    def is_eligible_for_retraining(self, approved_records: List[Dict[str, Any]]) -> bool:
        """
        Determines whether the approved feedback dataset is large enough to trigger
        retraining in Phase 12.
        """
        count = len(approved_records)
        eligible = count >= self.min_retraining_records
        logger.info(
            f"Retraining check: {count} approved records. "
            f"Required: {self.min_retraining_records}. Eligible: {eligible}"
        )
        return eligible

    def extract_approved_ids(self, review_history: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts the list of feedback IDs that are currently approved based on history.
        """
        # Find latest status for each unique feedback ID
        latest_status: Dict[str, str] = {}
        for entry in review_history:
            fid = entry.get("Feedback ID")
            status = entry.get("Review Status")
            if fid and status:
                latest_status[fid] = status.upper()

        approved_ids = [fid for fid, status in latest_status.items() if status == "APPROVED"]
        return approved_ids
