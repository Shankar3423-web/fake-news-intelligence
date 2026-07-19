import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("feature_selection_pipeline")

class SelectionProfiler:
    """
    Profiles the final selected dataset by analyzing distributions,
    null percentages, types, and the Pearson correlation matrix of selected features.
    """
    def __init__(self) -> None:
        self.profile: Dict[str, Any] = {}

    def profile_dataset(self, df_selected: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Generates profile information of the selected feature dataset.
        
        Args:
            df_selected: DataFrame containing the final selected dataset (including base cols).
            feature_columns: List of selected feature names.
            
        Returns:
            Dictionary containing the dataset profile.
        """
        logger.info("Profiling selected feature dataset...")
        
        feature_profiles = {}
        total_rows = len(df_selected)
        
        # 1. Feature-level profiles
        for col in feature_columns:
            if col in df_selected.columns:
                try:
                    series = df_selected[col]
                    null_count = int(series.isnull().sum())
                    null_pct = float(null_count / total_rows) if total_rows > 0 else 0.0
                    
                    # Percentiles
                    q25 = float(series.quantile(0.25))
                    q50 = float(series.quantile(0.50))
                    q75 = float(series.quantile(0.75))
                    
                    feature_profiles[col] = {
                        "dtype": str(series.dtype),
                        "null_count": null_count,
                        "null_percentage": round(null_pct, 4),
                        "distribution": {
                            "min": round(float(series.min()), 4),
                            "25%": round(q25, 4),
                            "50%": round(q50, 4),
                            "75%": round(q75, 4),
                            "max": round(float(series.max()), 4)
                        }
                    }
                except Exception as e:
                    logger.error(f"Failed to profile selected column '{col}': {e}")
                    feature_profiles[col] = {"error": str(e)}

        # 2. Pearson correlation matrix among selected features
        correlation_matrix = {}
        try:
            if len(feature_columns) > 1:
                # Compute pearson correlation
                corr_df = df_selected[feature_columns].corr(method="pearson")
                # Replace NaN values with 0.0 for JSON compatibility
                corr_df = corr_df.fillna(0.0)
                
                # Convert correlation DataFrame to nested dict format
                for col1 in corr_df.columns:
                    correlation_matrix[col1] = {}
                    for col2 in corr_df.index:
                        val = float(corr_df.loc[col2, col1])
                        correlation_matrix[col1][col2] = round(val, 4)
            else:
                logger.info("Fewer than 2 features; skipping correlation matrix.")
        except Exception as e:
            logger.error(f"Failed to calculate Pearson correlation matrix for selected features: {e}")
            correlation_matrix = {"error": str(e)}

        self.profile = {
            "dataset_summary": {
                "total_rows": total_rows,
                "total_selected_features": len(feature_columns)
            },
            "features": feature_profiles,
            "correlation_matrix": correlation_matrix
        }

        return self.profile

    def save(self, file_path: str) -> None:
        """Saves selection profile to JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=4)
            logger.info(f"Selection profile saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save selection profile to {file_path}: {e}")
            raise e
