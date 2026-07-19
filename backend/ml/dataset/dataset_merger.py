import logging
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

class DatasetMerger:
    """
    Merges multiple standardized DataFrames into a single, unified DataFrame.
    Verifies column alignments and ensures row preservation.
    """

    def __init__(self) -> None:
        pass

    def merge(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merges multiple DataFrames.
        
        Args:
            dataframes: A list of standardized pandas DataFrames.
            
        Returns:
            A single concatenated DataFrame.
            
        Raises:
            ValueError: If the input list is empty or if schemas are incompatible.
        """
        if not dataframes:
            error_msg = "Cannot merge an empty list of DataFrames."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Merging {len(dataframes)} datasets.")
        
        # Verify columns are aligned
        expected_cols = set(dataframes[0].columns)
        for i, df in enumerate(dataframes):
            df_cols = set(df.columns)
            if df_cols != expected_cols:
                error_msg = (
                    f"DataFrame at index {i} has columns {df_cols}, "
                    f"which does not match expected columns {expected_cols}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
                
        # Calculate expected row count
        expected_rows = sum(len(df) for df in dataframes)
        logger.info(f"Merging total expected rows: {expected_rows}")

        # Concatenate
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # Verify row preservation
        actual_rows = len(merged_df)
        if actual_rows != expected_rows:
            logger.warning(
                f"Row mismatch after merge! Expected: {expected_rows}, Actual: {actual_rows}"
            )
        else:
            logger.info(f"Merge successful. Consolidated shape: {merged_df.shape}")

        return merged_df
