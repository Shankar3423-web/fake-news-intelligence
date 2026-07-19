"""
training_executor.py
Executes the Phase 6 training pipeline adapted for Phase 12 candidate model training.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.retraining.retraining_config import RetrainingConfig

logger = logging.getLogger("retraining_pipeline")


class TrainingExecutor:
    """
    Trains a candidate model set on the merged dataset.

    Rather than re-invoking the full Phase 6 pipeline (which reads from fixed
    config paths), this executor directly reuses the lower-level training
    components (TrainerFactory, DatasetSplitter, etc.) so that it can train
    on the *merged* DataFrame passed in memory and save models to the
    candidate directory.
    """

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config

    def execute(
        self,
        merged_df: pd.DataFrame,
        feature_columns: List[str],
        training_config_path: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Trains classifiers on *merged_df* and saves artifacts to the
        candidate models directory.

        Args:
            merged_df:            Post-merge DataFrame containing features + label.
            feature_columns:      List of feature column names.
            training_config_path: Optional path to override the Phase 6 YAML config.

        Returns:
            Tuple ``(success, error_message, training_summary)`` where
            *training_summary* contains model summaries, durations, etc.
        """
        cfg_path = training_config_path or self._config.get_phase_config_path("training")

        logger.info(
            "TrainingExecutor: training candidate models on %d rows, %d features.",
            len(merged_df),
            len(feature_columns),
        )

        try:
            from ml.training.training_config import TrainingConfig
            from ml.training.dataset_splitter import DatasetSplitter
            from ml.training.trainer_factory import TrainerFactory
            from ml.training.training_profiler import TrainingProfiler

            training_cfg = TrainingConfig(cfg_path)

            candidate_dir = self._config.get_candidate_models_dir()
            os.makedirs(candidate_dir, exist_ok=True)

            # Split merged dataset
            splitter = DatasetSplitter(
                test_size=training_cfg.test_size,
                random_state=training_cfg.random_state,
                stratify=training_cfg.stratify,
                shuffle=training_cfg.shuffle,
            )
            X_train, X_test, y_train, y_test = splitter.split(
                merged_df, feature_columns
            )
            split_info = splitter.get_split_info()

            model_summaries: List[Dict[str, Any]] = []
            model_durations: Dict[str, float] = {}
            models_to_train = [
                "logistic_regression",
                "svm",
                "random_forest",
                "xgboost",
            ]

            for model_key in models_to_train:
                if not training_cfg.is_model_enabled(model_key):
                    logger.info("Model '%s' disabled — skipping.", model_key)
                    continue

                logger.info("Training candidate model: %s", model_key)
                hyperparams = training_cfg.get_model_hyperparameters(model_key)

                with TrainingProfiler.profile(
                    f"Candidate {model_key.upper()} Training"
                ) as benchmark:
                    trainer = TrainerFactory.create_trainer(model_key, hyperparams)
                    summary = trainer.train(X_train, y_train, split_info)
                    trainer.save(candidate_dir)

                model_durations[model_key] = benchmark.get("duration_sec", 0.0)
                model_summaries.append(summary)
                logger.info(
                    "Candidate '%s' trained in %.4fs.",
                    model_key,
                    model_durations[model_key],
                )

            training_summary: Dict[str, Any] = {
                "models_trained": [s.get("algorithm", "") for s in model_summaries],
                "model_summaries": model_summaries,
                "model_durations": model_durations,
                "split_info": split_info,
                "candidate_models_dir": candidate_dir,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
                "feature_count": len(feature_columns),
                "X_test": X_test,
                "y_test": y_test,
            }

            logger.info(
                "Candidate training complete. Models: %s",
                list(model_durations.keys()),
            )
            return True, "", training_summary

        except Exception as exc:  # noqa: BLE001
            msg = f"Candidate training failed: {exc}"
            logger.exception(msg)
            return False, msg, {}
