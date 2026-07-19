import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("feature_engineering_pipeline")

class FeatureProfiler:
    """
    Analyzes, profiles, and logs distributions and correlations for all engineered features.
    """
    def __init__(self) -> None:
        self.profile: Dict[str, Any] = {}

    def profile_dataset(self, df: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Creates a profiling dictionary containing:
        - Schema info (columns, data types)
        - Null counts & percentages
        - Quantile distributions (25%, 50%, 75%)
        - Pearson correlation matrix among dense features
        """
        logger.info("Profiling feature dataset...")
        
        feature_profiles = {}
        total_rows = len(df)
        
        # 1. Feature-level profiles
        for col in feature_columns:
            if col in df.columns:
                try:
                    series = df[col]
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
                            "25%": round(q25, 4),
                            "50%": round(q50, 4),
                            "75%": round(q75, 4)
                        }
                    }
                except Exception as e:
                    logger.error(f"Failed to profile column '{col}': {e}")
                    feature_profiles[col] = {"error": str(e)}
        
        # 2. Correlation Matrix calculation
        correlation_matrix = {}
        try:
            if len(feature_columns) > 1:
                # Compute pearson correlation
                corr_df = df[feature_columns].corr(method="pearson")
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
            logger.error(f"Failed to calculate Pearson correlation matrix: {e}")
            correlation_matrix = {"error": str(e)}
            
        self.profile = {
            "dataset_summary": {
                "total_rows": total_rows,
                "total_features": len(feature_columns)
            },
            "features": feature_profiles,
            "correlation_matrix": correlation_matrix
        }
        
        return self.profile

    def save(self, file_path: str) -> None:
        """
        Saves the generated dataset profile as a JSON file.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.profile, f, indent=4)
            logger.info(f"Feature profile saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save feature profile to {file_path}: {e}")
            raise e
