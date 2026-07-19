import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger("feature_selection_pipeline")

class SelectionStatistics:
    """
    Tracks, calculates, and saves statistics of the feature selection process,
    such as feature counts, reduction rates, and component execution times.
    """
    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {}

    def calculate_statistics(
        self,
        total_dense_in: int,
        total_sparse_in: int,
        selected_dense: List[str],
        selected_sparse: List[str],
        selector_runtimes: Dict[str, float],
        selector_counts: Dict[str, int],
        total_execution_time: float
    ) -> Dict[str, Any]:
        """
        Calculates and aggregates selection statistics.
        """
        total_in = total_dense_in + total_sparse_in
        total_out = len(selected_dense) + len(selected_sparse)
        reduction_rate = (1.0 - (total_out / total_in)) * 100.0 if total_in > 0 else 0.0

        self.stats = {
            "summary": {
                "total_input_features": total_in,
                "total_output_features": total_out,
                "reduction_percentage": round(reduction_rate, 2),
                "total_execution_time_seconds": round(total_execution_time, 2)
            },
            "by_feature_type": {
                "dense": {
                    "input": total_dense_in,
                    "output": len(selected_dense),
                    "reduction_percentage": round((1.0 - (len(selected_dense) / total_dense_in)) * 100.0, 2) if total_dense_in > 0 else 0.0
                },
                "sparse": {
                    "input": total_sparse_in,
                    "output": len(selected_sparse),
                    "reduction_percentage": round((1.0 - (len(selected_sparse) / total_sparse_in)) * 100.0, 2) if total_sparse_in > 0 else 0.0
                }
            },
            "selector_runtimes_seconds": {
                name: round(val, 4) for name, val in selector_runtimes.items()
            },
            "selector_selected_counts": selector_counts
        }

        logger.info(f"Selection statistics: Input={total_in} -> Output={total_out} ({reduction_rate:.2f}% reduction) in {total_execution_time:.2f}s.")
        return self.stats

    def save(self, file_path: str) -> None:
        """Saves selection statistics to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=4)
            logger.info(f"Selection statistics saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save selection statistics to {file_path}: {e}")
            raise e
