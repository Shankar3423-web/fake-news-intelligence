import os
import json
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("feature_engineering_pipeline")

class FeatureStatistics:
    """
    Collects, aggregates, and saves summary statistics for all engineered features.
    """
    def __init__(self) -> None:
        self.statistics: Dict[str, Any] = {}

    def calculate_statistics(
        self, 
        df: pd.DataFrame, 
        feature_columns: list, 
        execution_time: float
    ) -> Dict[str, Any]:
        """
        Calculates mean, std, min, max, and null count for each numerical feature column.
        """
        logger.info("Calculating feature summary statistics...")
        
        feature_metrics = {}
        total_records = len(df)
        
        for col in feature_columns:
            if col in df.columns:
                try:
                    series = df[col]
                    mean_val = float(series.mean())
                    std_val = float(series.std()) if len(series) > 1 else 0.0
                    min_val = float(series.min())
                    max_val = float(series.max())
                    null_count = int(series.isnull().sum())
                    
                    feature_metrics[col] = {
                        "mean": round(mean_val, 4),
                        "std": round(std_val, 4),
                        "min": round(min_val, 4),
                        "max": round(max_val, 4),
                        "null_count": null_count
                    }
                except Exception as e:
                    logger.error(f"Failed to calculate statistics for column '{col}': {e}")
                    feature_metrics[col] = {
                        "error": str(e)
                    }
            else:
                logger.warning(f"Feature column '{col}' not found in DataFrame for statistics.")
        
        self.statistics = {
            "total_records": total_records,
            "execution_time_seconds": round(execution_time, 2),
            "feature_metrics": feature_metrics
        }
        
        return self.statistics

    def save(self, file_path: str) -> None:
        """
        Saves the calculated statistics to a JSON file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.statistics, f, indent=4)
            logger.info(f"Feature statistics saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save feature statistics to {file_path}: {e}")
            raise e
