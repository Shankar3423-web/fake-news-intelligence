import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("admin_review_pipeline")

class StatisticsManager:
    """
    StatisticsManager aggregates review history metrics and saves them to a JSON file.
    """
    def __init__(self, stats_file_path: str) -> None:
        self.stats_file_path = stats_file_path

    def load_statistics(self) -> Dict[str, Any]:
        """
        Loads statistics from the JSON file. Returns defaults if file doesn't exist.
        """
        if not os.path.exists(self.stats_file_path):
            return {
                "total_reviews": 0,
                "approved_count": 0,
                "rejected_count": 0,
                "pending_count": 0,
                "approval_rate": 0.0,
                "Total Reviews": 0,
                "Approved Count": 0,
                "Rejected Count": 0,
                "Pending Count": 0,
                "Approval Rate": 0.0
            }

        try:
            with open(self.stats_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading admin review statistics: {e}")
            return {}

    def calculate_and_save(self, review_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes review statistics based on the LATEST decision for each unique Feedback ID,
        saves them to admin_review_statistics.json, and returns the stats.
        """
        # Determine latest status for each feedback ID
        latest_status: Dict[str, str] = {}
        for entry in review_history:
            fid = entry.get("Feedback ID")
            status = entry.get("Review Status")
            if fid and status:
                latest_status[fid] = status.upper()

        total_reviews = len(latest_status)
        approved_count = sum(1 for status in latest_status.values() if status == "APPROVED")
        rejected_count = sum(1 for status in latest_status.values() if status == "REJECTED")
        pending_count = sum(1 for status in latest_status.values() if status == "PENDING")

        approval_rate = 0.0
        if total_reviews > 0:
            approval_rate = approved_count / total_reviews

        stats = {
            "total_reviews": total_reviews,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "pending_count": pending_count,
            "approval_rate": approval_rate,
            "Total Reviews": total_reviews,
            "Approved Count": approved_count,
            "Rejected Count": rejected_count,
            "Pending Count": pending_count,
            "Approval Rate": approval_rate
        }

        os.makedirs(os.path.dirname(self.stats_file_path), exist_ok=True)
        try:
            with open(self.stats_file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)
            logger.info(f"Admin review statistics saved to {self.stats_file_path}")
        except Exception as e:
            logger.error(f"Failed to save admin review statistics: {e}")
            raise

        return stats
