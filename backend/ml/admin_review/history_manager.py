import os
import csv
import logging
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger("admin_review_pipeline")

class HistoryManager:
    """
    HistoryManager handles saving and loading the administrative review audit history.
    """
    def __init__(self, history_csv_path: str) -> None:
        self.history_csv_path = history_csv_path
        self.headers = ["Feedback ID", "Review Status", "Reviewer", "Review Notes", "Timestamp"]

    def save_history(
        self,
        feedback_id: str,
        status: str,
        reviewer: str,
        notes: str,
        timestamp: str
    ) -> None:
        """
        Appends a review record to the review history CSV.
        """
        file_exists = os.path.exists(self.history_csv_path) and os.path.getsize(self.history_csv_path) > 0
        os.makedirs(os.path.dirname(self.history_csv_path), exist_ok=True)

        try:
            with open(self.history_csv_path, mode="a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.headers)
                writer.writerow([
                    feedback_id,
                    status.upper(),
                    reviewer,
                    notes or "",
                    timestamp
                ])
            logger.info(f"Saved review transaction for feedback ID {feedback_id} to review history.")
        except Exception as e:
            logger.error(f"Error saving review transaction to history: {e}")
            raise

    def load_history(self) -> List[Dict[str, Any]]:
        """
        Loads all review history transactions.
        """
        if not os.path.exists(self.history_csv_path):
            return []

        try:
            df = pd.read_csv(self.history_csv_path)
            df = df.where(pd.notnull(df), None)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error loading review history: {e}")
            return []
