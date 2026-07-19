import os
import json
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatasetStatistics:
    """
    Generates high-level metadata statistics summarizing the pipeline execution
    and final dataset sizes. Saves results to dataset_statistics.json.
    """

    def __init__(self) -> None:
        pass

    def generate_statistics(
        self,
        raw_counts: Dict[str, int],
        cleaned_counts: Dict[str, int],
        final_count: int,
        duplicates_removed: int,
        missing_mandatory_dropped: int,
        real_count: int,
        fake_count: int,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Creates and saves high-level statistics summary.
        
        Args:
            raw_counts: Raw row counts mapped by file label.
            cleaned_counts: Cleaned row counts mapped by file label.
            final_count: Final master row count.
            duplicates_removed: Number of duplicate rows removed.
            missing_mandatory_dropped: Number of rows dropped due to missing mandatory fields.
            real_count: Number of real articles.
            fake_count: Number of fake articles.
            output_path: Path to save the dataset_statistics.json.
            
        Returns:
            A dictionary of summary statistics.
        """
        logger.info(f"Generating high-level dataset statistics.")
        
        # Calculate rates
        total_raw = sum(raw_counts.values())
        raw_to_final_pct = float(round((final_count / total_raw) * 100, 4)) if total_raw > 0 else 0.0
        
        stats = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "execution_status": "Success",
            "pipeline_summary": {
                "total_raw_rows_loaded": total_raw,
                "total_rows_dropped_missing_mandatory": missing_mandatory_dropped,
                "total_rows_removed_duplicates": duplicates_removed,
                "total_final_rows_preserved": final_count,
                "yield_percentage": raw_to_final_pct
            },
            "source_breakdown": {
                "raw": raw_counts,
                "cleaned": cleaned_counts
            },
            "class_distribution": {
                "real_news_count": real_count,
                "fake_news_count": fake_count,
                "real_ratio": float(round((real_count / final_count) * 100, 4)) if final_count > 0 else 0.0,
                "fake_ratio": float(round((fake_count / final_count) * 100, 4)) if final_count > 0 else 0.0
            },
            "file_metadata": {
                "master_dataset_file": "ml/dataset/processed/master_dataset_v1.csv",
                "master_dataset_size_bytes": 0
            }
        }

        # Attempt to get file size of the master dataset if it exists
        master_file_path = os.path.join(
            os.path.dirname(os.path.dirname(output_path)), 
            "processed", 
            "master_dataset_v1.csv"
        )
        if os.path.exists(master_file_path):
            stats["file_metadata"]["master_dataset_size_bytes"] = os.path.getsize(master_file_path)

        # Save stats to disk
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)
            logger.info(f"Saved dataset statistics to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset statistics to {output_path}. Error: {str(e)}")

        return stats
