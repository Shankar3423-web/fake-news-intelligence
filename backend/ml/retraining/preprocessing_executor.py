"""
preprocessing_executor.py
Delegates to the existing Phase 3 preprocessing pipeline for Phase 12.
"""
import logging
import os
import sys
from typing import Tuple

logger = logging.getLogger("retraining_pipeline")


class PreprocessingExecutor:
    """
    Executes the existing Phase 3 NLP Preprocessing pipeline via its
    public :func:`run_preprocessing_pipeline` entry-point.

    This executor follows the *Adapter* pattern: it wraps the existing
    pipeline function with retraining-specific logging and error handling
    without duplicating any preprocessing logic.
    """

    def __init__(self, config_path: str) -> None:
        """
        Args:
            config_path: Path to the preprocessing YAML config
                         (``config/preprocessing_config.yaml``).
        """
        self._config_path = config_path

    def execute(self) -> Tuple[bool, str]:
        """
        Runs Phase 3 preprocessing pipeline.

        Returns
        -------
        ``(success, error_message)`` — ``error_message`` is empty on success.
        """
        logger.info(
            "PreprocessingExecutor: delegating to Phase 3 pipeline "
            "(config=%s).",
            self._config_path,
        )

        try:
            # Import here to avoid circular imports at module load time
            from ml.preprocessing.preprocessing_pipeline import (
                run_preprocessing_pipeline,
            )

            success = run_preprocessing_pipeline(self._config_path)
            if success:
                logger.info("Phase 3 preprocessing pipeline completed successfully.")
                return True, ""
            else:
                msg = "Phase 3 preprocessing pipeline returned False (validation failure)."
                logger.error(msg)
                return False, msg

        except Exception as exc:  # noqa: BLE001
            msg = f"Phase 3 preprocessing pipeline raised an exception: {exc}"
            logger.exception(msg)
            return False, msg
