import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("prediction_pipeline")

class PredictionStatistics:
    """
    Tracks and updates overall statistics for all prediction engine executions.
    Maintains a running average of performance and confidence metrics.
    """
    def __init__(self, stats_file_path: str = "ml/prediction/statistics/prediction_statistics.json") -> None:
        self.stats_file_path = stats_file_path
        os.makedirs(os.path.dirname(self.stats_file_path), exist_ok=True)

    def load_statistics(self) -> Dict[str, Any]:
        """Loads statistics from file, returns defaults if file does not exist."""
        if not os.path.exists(self.stats_file_path):
            return {
                "total_predictions": 0,
                "average_prediction_time_ms": 0.0,
                "average_memory_usage_mb": 0.0,
                "average_throughput_sps": 0.0,
                "average_confidence": 0.0,
                "model_used": "none"
            }
            
        try:
            with open(self.stats_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to parse statistics file: {e}. Reinitializing stats.")
            return {
                "total_predictions": 0,
                "average_prediction_time_ms": 0.0,
                "average_memory_usage_mb": 0.0,
                "average_throughput_sps": 0.0,
                "average_confidence": 0.0,
                "model_used": "none"
            }

    def update_statistics(
        self,
        prediction_time_ms: float,
        memory_usage_mb: float,
        throughput_sps: float,
        confidence: float,
        model_used: str
    ) -> Dict[str, Any]:
        """
        Updates the running statistics with the metrics from a new prediction.
        """
        logger.info("Updating prediction engine statistics...")
        stats = self.load_statistics()
        
        n = stats["total_predictions"]
        new_n = n + 1
        
        # Calculate running averages
        avg_time = ((stats["average_prediction_time_ms"] * n) + prediction_time_ms) / new_n
        avg_mem = ((stats["average_memory_usage_mb"] * n) + memory_usage_mb) / new_n
        avg_tp = ((stats["average_throughput_sps"] * n) + throughput_sps) / new_n
        avg_conf = ((stats["average_confidence"] * n) + confidence) / new_n
        
        updated_stats = {
            "total_predictions": new_n,
            "average_prediction_time_ms": float(round(avg_time, 4)),
            "average_memory_usage_mb": float(round(avg_mem, 4)),
            "average_throughput_sps": float(round(avg_tp, 2)),
            "average_confidence": float(round(avg_conf, 4)),
            "model_used": model_used
        }
        
        try:
            with open(self.stats_file_path, "w", encoding="utf-8") as f:
                json.dump(updated_stats, f, indent=4)
            logger.info(f"Statistics updated successfully. Total runs: {new_n}")
        except Exception as e:
            logger.error(f"Failed to save statistics to {self.stats_file_path}: {e}")
            raise e
            
        return updated_stats
