import os
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger("verification_pipeline")

class HistoryManager:
    """
    HistoryManager logs each verification transaction to a persistent CSV database.
    """
    def __init__(self, history_csv_path: str = "ml/verification/history/verification_history.csv") -> None:
        self.history_csv_path = history_csv_path
        os.makedirs(os.path.dirname(self.history_csv_path), exist_ok=True)

    def save_history(
        self,
        input_hash: str,
        prediction: str,
        verification_status: str,
        evidence_score: float,
        similarity: float,
        providers: str
    ) -> None:
        """
        Appends a run entry to the verification history CSV file.
        """
        row_data = {
            "Timestamp": datetime.now().isoformat(),
            "Input Hash": input_hash,
            "Prediction": prediction,
            "Verification Status": verification_status,
            "Evidence Score": round(evidence_score, 4),
            "Similarity": round(similarity, 4),
            "Providers": providers
        }

        try:
            df_row = pd.DataFrame([row_data])
            header = not os.path.exists(self.history_csv_path)
            df_row.to_csv(self.history_csv_path, mode="a", index=False, header=header)
            logger.info(f"Successfully appended record to verification history CSV: {self.history_csv_path}")
        except Exception as e:
            logger.error(f"Failed to append record to history CSV at {self.history_csv_path}: {e}")
