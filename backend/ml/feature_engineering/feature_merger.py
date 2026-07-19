import pandas as pd
import logging
from typing import List

logger = logging.getLogger("feature_engineering_pipeline")

class FeatureMerger:
    """
    Merges base columns (id, label, cleaned_text) and engineered dense feature groups
    into a single tabular DataFrame.
    """
    def __init__(self) -> None:
        pass

    def merge_features(self, base_df: pd.DataFrame, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merges base columns with a list of feature DataFrames column-wise.
        Checks for row count matches to ensure data integrity.
        """
        logger.info("Merging base columns with engineered feature dataframes...")
        
        # Verify required columns exist in the base dataframe
        required_base_cols = ["id", "label", "cleaned_text"]
        missing_cols = [col for col in required_base_cols if col not in base_df.columns]
        if missing_cols:
            logger.error(f"Base DataFrame is missing required columns: {missing_cols}")
            raise ValueError(f"Base DataFrame is missing required columns: {missing_cols}")
            
        # Create a copy with only the required columns
        merged_df = base_df[required_base_cols].copy().reset_index(drop=True)
        
        for idx, df in enumerate(feature_dfs):
            if df.empty:
                continue
                
            if len(df) != len(merged_df):
                logger.error(f"Row count mismatch in feature dataframe index {idx}. "
                             f"Base rows: {len(merged_df)}, feature rows: {len(df)}")
                raise ValueError("Cannot merge dataframes with mismatched row counts.")
                
            # Align indices and concatenate
            df_reset = df.reset_index(drop=True)
            merged_df = pd.concat([merged_df, df_reset], axis=1)
            
        logger.info(f"Features successfully merged. Output dataset shape: {merged_df.shape}")
        return merged_df
