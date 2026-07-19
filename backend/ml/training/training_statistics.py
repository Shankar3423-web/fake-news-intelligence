import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("model_training_pipeline")

class TrainingStatistics:
    """
    Collects, aggregates, and saves model training execution statistics.
    Produces training_statistics.json.
    """
    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {
            "dataset_size_bytes": 0,
            "feature_count": 0,
            "training_rows": 0,
            "testing_rows": 0,
            "training_duration_sec": 0.0,
            "memory_used_mb": 0.0,
            "per_model_duration_sec": {},
            "total_models_trained": 0,
            "average_training_time_sec": 0.0
        }

    def generate(
        self,
        dataset_path: str,
        feature_count: int,
        training_rows: int,
        testing_rows: int,
        total_pipeline_duration: float,
        peak_memory_used_mb: float,
        model_durations: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Populates the statistics dictionary.
        """
        # Get file size in bytes
        dataset_size = 0
        if os.path.exists(dataset_path):
            dataset_size = os.path.getsize(dataset_path)

        total_models = len(model_durations)
        avg_time = 0.0
        if total_models > 0:
            avg_time = round(sum(model_durations.values()) / total_models, 4)

        self.stats = {
            "dataset_size_bytes": dataset_size,
            "feature_count": feature_count,
            "training_rows": training_rows,
            "testing_rows": testing_rows,
            "training_duration_sec": round(total_pipeline_duration, 4),
            "memory_used_mb": round(peak_memory_used_mb, 2),
            "per_model_duration_sec": {k: round(v, 4) for k, v in model_durations.items()},
            "total_models_trained": total_models,
            "average_training_time_sec": avg_time
        }
        
        logger.info(f"Generated training statistics: Models={total_models}, AvgTime={avg_time}s")
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
            logger.info(f"Saved training statistics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save training statistics to {filepath}: {e}")
            raise
