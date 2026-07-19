import os
import csv
import logging
import pandas as pd
from typing import Dict, Any, List, Set

logger = logging.getLogger("admin_review_pipeline")

class ApprovedDatasetManager:
    """
    ApprovedDatasetManager manages the curated dataset of approved feedback records
    suitable for future model retraining.
    """
    def __init__(self, approved_csv_path: str) -> None:
        self.approved_csv_path = approved_csv_path

    def sync_approved_dataset(
        self,
        latest_review_states: Dict[str, str],
        feedback_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Synchronizes the approved feedback dataset based on the latest review decisions.
        Only feedback records whose latest review decision is 'APPROVED' are included.
        Overwrites approved_feedback.csv with the updated curated dataset.
        """
        approved_records = []
        
        # Build set of approved feedback IDs
        approved_ids: Set[str] = {
            fid for fid, status in latest_review_states.items() if status == "APPROVED"
        }

        # Filter feedback records to only include APPROVED ones
        for record in feedback_records:
            fid = str(record.get("Feedback ID", "")).strip()
            if fid in approved_ids:
                approved_records.append(record)

        # Write to approved_feedback.csv
        os.makedirs(os.path.dirname(self.approved_csv_path), exist_ok=True)
        try:
            if approved_records:
                df = pd.DataFrame(approved_records)
                # Keep column order consistent with original feedback history CSV
                # Columns: Feedback ID, Timestamp, Prediction, Verification, Decision, Feedback, Comment
                col_order = ["Feedback ID", "Timestamp", "Prediction", "Verification", "Decision", "Feedback", "Comment"]
                # Filter to only existing columns in order, or create missing if any
                cols_to_use = [col for col in col_order if col in df.columns]
                df = df[cols_to_use]
                df.to_csv(self.approved_csv_path, index=False, encoding="utf-8")
            else:
                # If there are no approved records, create an empty CSV with headers
                with open(self.approved_csv_path, mode="w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Feedback ID", "Timestamp", "Prediction", "Verification", "Decision", "Feedback", "Comment"])

            logger.info(f"Synchronized approved dataset at {self.approved_csv_path}. Total approved: {len(approved_records)}")
        except Exception as e:
            logger.error(f"Failed to synchronize approved dataset at {self.approved_csv_path}: {e}")
            raise

        return approved_records

    def load_approved_dataset(self) -> List[Dict[str, Any]]:
        """
        Loads the current approved feedback records from the approved dataset CSV.
        """
        if not os.path.exists(self.approved_csv_path):
            return []

        try:
            df = pd.read_csv(self.approved_csv_path)
            df = df.where(pd.notnull(df), None)
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error loading approved dataset from {self.approved_csv_path}: {e}")
            return []
