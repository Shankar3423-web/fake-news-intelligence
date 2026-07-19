import os
import logging
import pandas as pd
from typing import Any

logger = logging.getLogger("feedback_pipeline")

class HistoryManager:
    """
    HistoryManager logs each feedback transaction to a persistent CSV database.
    """
    def __init__(self, history_csv_path: str = "ml/feedback/history/feedback_history.csv") -> None:
        self.history_csv_path = history_csv_path
        os.makedirs(os.path.dirname(self.history_csv_path), exist_ok=True)

    def save_history(
        self,
        feedback_id: str,
        timestamp: str,
        prediction: Any,
        verification: str,
        decision: str,
        feedback: str,
        comment: str
    ) -> None:
        """
        Appends a run entry to the feedback history CSV file.
        """
        row_data = {
            "Feedback ID": feedback_id,
            "Timestamp": timestamp,
            "Prediction": prediction,
            "Verification": verification,
            "Decision": decision,
            "Feedback": feedback,
            "Comment": comment
        }

        try:
            df_row = pd.DataFrame([row_data])
            header = not os.path.exists(self.history_csv_path)
            df_row.to_csv(self.history_csv_path, mode="a", index=False, header=header, encoding="utf-8")
            logger.info(f"Successfully appended record to feedback history CSV: {self.history_csv_path}")
        except Exception as e:
            logger.error(f"Failed to append record to history CSV at {self.history_csv_path}: {e}")
