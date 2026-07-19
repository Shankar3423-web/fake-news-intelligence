"""
approved_feedback_loader.py
Loads and validates administrator-approved feedback records from Phase 11.
"""
import logging
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from ml.retraining.retraining_config import RetrainingConfig

logger = logging.getLogger("retraining_pipeline")


class ApprovedFeedbackLoader:
    """
    Loads approved feedback records produced by Phase 11 (Admin Review System).

    Responsibilities
    ----------------
    - Read the approved_feedback.csv file.
    - Validate schema: required columns present, non-empty data.
    - Validate labels: Decision column maps to a valid integer label.
    - Return a clean :class:`~pandas.DataFrame` for downstream merging.
    """

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config
        self._csv_path: str = config.get_approved_feedback_path()
        self._label_map: Dict[str, int] = config.feedback_label_mapping

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def load(self) -> Tuple[bool, str, pd.DataFrame]:
        """
        Loads and validates approved feedback records.

        Returns
        -------
        Tuple of (success, error_message, dataframe).
        On success *error_message* is an empty string and *dataframe* contains
        the approved records with an added ``label`` integer column.
        On failure *dataframe* is an empty DataFrame.
        """
        logger.info("Loading approved feedback from PostgreSQL database...")

        try:
            from app.database.session import SessionLocal
            from app.database.models.feedback import Feedback
            
            with SessionLocal() as db:
                # Only load feedbacks that have been explicitly APPROVED
                approved_feedbacks = db.query(Feedback).filter(Feedback.status == "APPROVED").all()
                
                if not approved_feedbacks:
                    msg = "No approved records found in the database."
                    logger.warning(msg)
                    return False, msg, pd.DataFrame()
                    
                records = []
                for f in approved_feedbacks:
                    if not f.prediction:
                        continue
                        
                    # Calculate ground truth based on model prediction + user's correction
                    predicted = f.prediction.predicted_label.upper()
                    
                    if f.is_correct:
                        # User agrees, so the original prediction was the true label
                        decision = predicted
                    else:
                        # User disagrees, so the true label is the opposite
                        decision = "REAL" if predicted == "FAKE" else "FAKE"
                        
                    label_int = self._label_map.get(decision.upper(), None)
                    
                    if label_int is not None:
                        records.append({
                            "Feedback ID": str(f.id),
                            "label": label_int
                        })
                        
                if not records:
                    msg = "No valid records could be processed from approved feedback."
                    logger.warning(msg)
                    return False, msg, pd.DataFrame()
                    
                df = pd.DataFrame(records)
                logger.info(
                    "Loaded %d approved feedback records from database (label distribution: %s).",
                    len(df),
                    df["label"].value_counts().to_dict(),
                )
                return True, "", df
                
        except Exception as exc:
            msg = f"Failed to read approved feedback from database: {exc}"
            logger.error(msg)
            return False, msg, pd.DataFrame()

    # ── Properties ────────────────────────────────────────────────────────────────────────────────
    @property
    def csv_path(self) -> str:
        """Absolute path to the approved feedback CSV."""
        return self._csv_path
