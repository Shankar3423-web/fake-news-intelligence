"""
statistics_manager.py
Generates and persists retraining pipeline execution statistics for Phase 12.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List

logger = logging.getLogger("retraining_pipeline")


class RetrainingStatisticsManager:
    """
    Collects and persists retraining pipeline execution statistics.

    Produced artifact: ``retraining_statistics.json``
    """

    def __init__(self) -> None:
        self._stats: Dict[str, Any] = {}

    def generate(
        self,
        *,
        run_id: str,
        approved_records: int,
        existing_rows: int,
        merged_rows: int,
        feature_count: int,
        training_rows: int,
        testing_rows: int,
        pipeline_duration_sec: float,
        candidate_metrics: Dict[str, Any],
        production_metrics: Dict[str, Any],
        decision: str,
        model_durations: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Populates the statistics dictionary.

        Args:
            run_id:               Unique retraining run ID.
            approved_records:     Approved feedback records merged.
            existing_rows:        Pre-merge training dataset size.
            merged_rows:          Post-merge training dataset size.
            feature_count:        Feature dimensions.
            training_rows:        Training split size.
            testing_rows:         Test split size.
            pipeline_duration_sec: Wall-clock time for the full pipeline.
            candidate_metrics:    Best candidate model evaluation metrics.
            production_metrics:   Current production model metrics at comparison.
            decision:             ``"PROMOTED"`` or ``"REJECTED"``.
            model_durations:      Per-model training durations (seconds).

        Returns:
            The generated statistics dictionary.
        """
        total_models = len(model_durations)
        avg_training_time = (
            round(sum(model_durations.values()) / total_models, 4)
            if total_models > 0
            else 0.0
        )

        self._stats = {
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset": {
                "approved_records_merged": approved_records,
                "pre_merge_rows": existing_rows,
                "post_merge_rows": merged_rows,
                "delta_rows": max(0, merged_rows - existing_rows),
                "feature_count": feature_count,
                "training_split": training_rows,
                "testing_split": testing_rows,
            },
            "training": {
                "total_models_trained": total_models,
                "per_model_duration_sec": {
                    k: round(v, 4) for k, v in model_durations.items()
                },
                "average_training_time_sec": avg_training_time,
            },
            "evaluation": {
                "candidate_metrics": candidate_metrics,
                "production_metrics_at_comparison": production_metrics,
            },
            "pipeline": {
                "total_duration_sec": round(pipeline_duration_sec, 4),
                "decision": decision,
            },
        }

        logger.info(
            "Retraining statistics generated: run_id=%s, decision=%s, "
            "duration=%.2fs.",
            run_id,
            decision,
            pipeline_duration_sec,
        )
        return self._stats

    def save(self, filepath: str) -> None:
        """Serialises statistics to *filepath* (JSON)."""
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self._stats, fh, indent=4, default=str)
        logger.info("Retraining statistics saved to: %s", filepath)

    @property
    def stats(self) -> Dict[str, Any]:
        """Returns the current statistics dictionary."""
        return dict(self._stats)
