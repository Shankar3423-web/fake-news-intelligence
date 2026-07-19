import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("feedback_pipeline")

class StatisticsManager:
    """
    StatisticsManager tracks, loads, and updates aggregate metrics for feedback.
    """
    def __init__(self, stats_file_path: str = "ml/feedback/statistics/feedback_statistics.json") -> None:
        self.stats_file_path = stats_file_path
        os.makedirs(os.path.dirname(self.stats_file_path), exist_ok=True)

    def load_statistics(self) -> Dict[str, Any]:
        """
        Loads statistics from file, or returns defaults if the file does not exist.
        """
        if not os.path.exists(self.stats_file_path):
            return self.get_defaults()
        
        try:
            with open(self.stats_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load statistics from {self.stats_file_path}: {e}. Returning defaults.")
            return self.get_defaults()

    def get_defaults(self) -> Dict[str, Any]:
        """Returns default dictionary structure for statistics."""
        return {
            "total_feedback": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "unsure_count": 0,
            "average_prediction_confidence": 0.0,
            "average_similarity": 0.0,
            "average_evidence_score": 0.0,
            "feedback_distribution": {
                "Correct": 0.0,
                "Incorrect": 0.0,
                "Unsure": 0.0
            }
        }

    def update_statistics(
        self,
        feedback_value: str,
        prediction_confidence: float,
        similarity_score: float,
        evidence_score: float
    ) -> Dict[str, Any]:
        """
        Updates statistics with the inputs from the current feedback and saves to file.
        """
        stats = self.load_statistics()

        # Update counts
        stats["total_feedback"] += 1
        
        fb_val = feedback_value.strip().capitalize()
        if fb_val == "Correct":
            stats["correct_count"] += 1
        elif fb_val == "Incorrect":
            stats["incorrect_count"] += 1
        elif fb_val == "Unsure":
            stats["unsure_count"] += 1

        total = stats["total_feedback"]

        # Calculate percentages for distribution
        stats["feedback_distribution"] = {
            "Correct": round(stats["correct_count"] / total, 4) if total > 0 else 0.0,
            "Incorrect": round(stats["incorrect_count"] / total, 4) if total > 0 else 0.0,
            "Unsure": round(stats["unsure_count"] / total, 4) if total > 0 else 0.0
        }

        # Running average prediction confidence
        old_avg_conf = stats.get("average_prediction_confidence", 0.0)
        stats["average_prediction_confidence"] = round(old_avg_conf + (prediction_confidence - old_avg_conf) / total, 4)

        # Running average similarity score
        old_avg_sim = stats.get("average_similarity", 0.0)
        stats["average_similarity"] = round(old_avg_sim + (similarity_score - old_avg_sim) / total, 4)

        # Running average evidence score
        old_avg_ev = stats.get("average_evidence_score", 0.0)
        stats["average_evidence_score"] = round(old_avg_ev + (evidence_score - old_avg_ev) / total, 4)

        # Write to file
        try:
            with open(self.stats_file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully updated feedback statistics at {self.stats_file_path}")
        except Exception as e:
            logger.error(f"Failed to write statistics file at {self.stats_file_path}: {e}")

        return stats
