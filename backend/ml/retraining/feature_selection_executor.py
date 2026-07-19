"""
feature_selection_executor.py
Delegates to the existing Phase 5 Feature Selection pipeline for Phase 12.
"""
import logging
from typing import Tuple

logger = logging.getLogger("retraining_pipeline")


class FeatureSelectionExecutor:
    """
    Executes the existing Phase 5 Feature Selection pipeline.

    Wraps :func:`~ml.feature_selection.feature_selection_pipeline.run_feature_selection_pipeline`
    following the Adapter pattern.
    """

    def __init__(self, config_path: str) -> None:
        """
        Args:
            config_path: Path to the feature selection YAML config.
        """
        self._config_path = config_path

    def execute(self) -> Tuple[bool, str]:
        """
        Runs the Phase 5 Feature Selection pipeline.

        Returns
        -------
        ``(success, error_message)`` — empty string on success.
        """
        logger.info(
            "FeatureSelectionExecutor: delegating to Phase 5 pipeline "
            "(config=%s).",
            self._config_path,
        )

        try:
            from ml.feature_selection.feature_selection_pipeline import (
                run_feature_selection_pipeline,
            )

            success = run_feature_selection_pipeline(self._config_path)
            if success:
                logger.info(
                    "Phase 5 feature selection pipeline completed successfully."
                )
                return True, ""
            else:
                msg = (
                    "Phase 5 feature selection pipeline returned False "
                    "(validation failure)."
                )
                logger.error(msg)
                return False, msg

        except Exception as exc:  # noqa: BLE001
            msg = f"Phase 5 feature selection pipeline raised an exception: {exc}"
            logger.exception(msg)
            return False, msg
