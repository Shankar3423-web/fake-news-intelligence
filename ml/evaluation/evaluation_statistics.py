import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("model_evaluation_pipeline")

class EvaluationStatistics:
    """
    Collects, aggregates, and saves model evaluation execution statistics.
    Produces evaluation_statistics.json.
    """
    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {}

    def generate(
        self,
        dataset_path: str,
        feature_count: int,
        dataset_size_rows: int,
        pipeline_duration: float,
        best_model_key: str,
        model_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates and formats statistical metrics across all evaluated classifiers.
        """
        # Get file size in bytes
        dataset_size_bytes = 0
        if os.path.exists(dataset_path):
            dataset_size_bytes = os.path.getsize(dataset_path)

        total_models = len(model_results)
        
        pred_times = {}
        memory_usages = {}
        for key, res in model_results.items():
            pred_times[key] = res["metrics"]["prediction_time_sec"]
            memory_usages[key] = res["metrics"]["memory_used_mb"]

        avg_pred_time = 0.0
        if total_models > 0:
            avg_pred_time = round(sum(pred_times.values()) / total_models, 4)

        self.stats = {
            "dataset_size_bytes": dataset_size_bytes,
            "dataset_rows": dataset_size_rows,
            "feature_count": feature_count,
            "prediction_times": pred_times,
            "average_prediction_time_sec": avg_pred_time,
            "memory_usages": memory_usages,
            "total_models_evaluated": total_models,
            "best_model": best_model_key,
            "evaluation_pipeline_duration_sec": round(pipeline_duration, 4)
        }
        
        logger.info(f"Generated evaluation statistics. Total models={total_models}, Best model='{best_model_key}'")
        return self.stats

    def save(self, filepath: str) -> None:
        """
        Saves the compiled statistics to a JSON file.
        """
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=4)
            logger.info(f"Saved evaluation statistics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save evaluation statistics: {e}")
            raise
