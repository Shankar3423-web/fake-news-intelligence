import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("verification_pipeline")

class StatisticsManager:
    """
    StatisticsManager tracks and updates aggregate execution metrics
    (run counts, status rates, average similarity, and cache hit rates)
    across all Phase 9 pipeline invocations.
    """
    def __init__(self, stats_file_path: str = "ml/verification/statistics/verification_statistics.json") -> None:
        self.stats_file_path = stats_file_path
        os.makedirs(os.path.dirname(self.stats_file_path), exist_ok=True)

    def load_statistics(self) -> Dict[str, Any]:
        """
        Loads statistics from file, or returns defaults if file does not exist.
        """
        if not os.path.exists(self.stats_file_path):
            return {
                "total_verifications": 0,
                "verified_count": 0,
                "partially_verified_count": 0,
                "not_verified_count": 0,
                "conflicting_count": 0,
                "verification_success_rate": 0.0,
                "average_similarity": 0.0,
                "average_api_time_seconds": 0.0,
                "average_provider_response_count": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0
            }
        
        try:
            with open(self.stats_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load statistics from {self.stats_file_path}: {e}. Returning defaults.")
            return {
                "total_verifications": 0,
                "verified_count": 0,
                "partially_verified_count": 0,
                "not_verified_count": 0,
                "conflicting_count": 0,
                "verification_success_rate": 0.0,
                "average_similarity": 0.0,
                "average_api_time_seconds": 0.0,
                "average_provider_response_count": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0
            }

    def update_statistics(
        self,
        verification_status: str,
        similarity_score: float,
        api_time_sec: float,
        response_count: int,
        cache_hit: bool
    ) -> Dict[str, Any]:
        """
        Updates statistics with the outcomes of the current run and saves to file.
        """
        stats = self.load_statistics()

        # Update counts
        stats["total_verifications"] += 1
        
        normalized_status = verification_status.upper()
        if normalized_status == "VERIFIED":
            stats["verified_count"] += 1
        elif normalized_status == "PARTIALLY VERIFIED":
            stats["partially_verified_count"] += 1
        elif normalized_status == "NOT VERIFIED":
            stats["not_verified_count"] += 1
        elif normalized_status == "CONFLICTING":
            stats["conflicting_count"] += 1

        # Success rate = verified / total
        stats["verification_success_rate"] = round(stats["verified_count"] / stats["total_verifications"], 4)

        # Running average similarity
        n = stats["total_verifications"]
        old_avg_sim = stats["average_similarity"]
        stats["average_similarity"] = round(old_avg_sim + (similarity_score - old_avg_sim) / n, 4)

        # Running average API latency
        old_avg_time = stats["average_api_time_seconds"]
        stats["average_api_time_seconds"] = round(old_avg_time + (api_time_sec - old_avg_time) / n, 4)

        # Running average articles returned
        old_avg_resp = stats["average_provider_response_count"]
        stats["average_provider_response_count"] = round(old_avg_resp + (response_count - old_avg_resp) / n, 4)

        # Cache stats
        if cache_hit:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1

        total_cache_reqs = stats["cache_hits"] + stats["cache_misses"]
        if total_cache_reqs > 0:
            stats["cache_hit_rate"] = round(stats["cache_hits"] / total_cache_reqs, 4)

        try:
            with open(self.stats_file_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully updated verification statistics at {self.stats_file_path}")
        except Exception as e:
            logger.error(f"Failed to write statistics file at {self.stats_file_path}: {e}")

        return stats
