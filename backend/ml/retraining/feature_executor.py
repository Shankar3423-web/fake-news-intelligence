"""
feature_executor.py
Delegates to the existing Phase 4 Feature Engineering pipeline for Phase 12.
"""
import logging
from typing import Tuple

logger = logging.getLogger("retraining_pipeline")


class FeatureExecutor:
    """
    Executes the existing Phase 4 Feature Engineering pipeline.

    Wraps :func:`~ml.feature_engineering.feature_pipeline.run_feature_engineering_pipeline`
    following the Adapter pattern to avoid code duplication.
    """

    def __init__(self, config_path: str) -> None:
        """
        Args:
            config_path: Path to the feature engineering YAML config.
        """
        self._config_path = config_path

    def execute(self) -> Tuple[bool, str]:
        """
        Runs the Phase 4 Feature Engineering pipeline.

        Returns
        -------
        ``(success, error_message)`` — empty string on success.
        """
        logger.info(
            "FeatureExecutor: delegating to Phase 4 pipeline "
            "(config=%s).",
            self._config_path,
        )

        try:
            from ml.feature_engineering.feature_pipeline import (
                run_feature_engineering_pipeline,
            )

            success = run_feature_engineering_pipeline(self._config_path)
            if success:
                logger.info(
                    "Phase 4 feature engineering pipeline completed successfully."
                )
                return True, ""
            else:
                msg = (
                    "Phase 4 feature engineering pipeline returned False "
                    "(validation failure)."
                )
                logger.error(msg)
                return False, msg

        except Exception as exc:  # noqa: BLE001
            msg = f"Phase 4 feature engineering pipeline raised an exception: {exc}"
            logger.exception(msg)
            return False, msg
