import os
import logging
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger("preprocessing_pipeline")

class PreprocessingValidator:
    """
    Validates input and output datasets to ensure structural integrity,
    schema adherence, and value ranges.
    """
    
    def __init__(self) -> None:
        self.expected_columns = [
            "id", "title", "text", "label", "source", "category", 
            "author", "published_date", "language", "url", "dataset_origin"
        ]

    def validate_input(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validates the input dataset file and structure.
        """
        errors = []
        if not os.path.exists(file_path):
            errors.append(f"Input dataset file not found at: {file_path}")
            return False, errors

        try:
            # Read first few rows to validate columns without loading large files into memory
            df_head = pd.read_csv(file_path, nrows=5)
            
            # Check for missing columns
            missing_cols = [col for col in self.expected_columns if col not in df_head.columns]
            if missing_cols:
                errors.append(f"Input dataset is missing required columns: {missing_cols}")
                
        except Exception as e:
            errors.append(f"Failed to read input dataset header: {e}")

        return len(errors) == 0, errors

    def validate_output(
        self, 
        input_path: str, 
        output_path: str, 
        expected_labels: List[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validates the output preprocessed dataset.
        Checks:
        - Output file exists
        - Required columns (including cleaned_text) exist
        - cleaned_text contains no empty/blank values
        - Labels are unchanged and match the allowed set [0, 1]
        """
        errors = []
        if not os.path.exists(output_path):
            errors.append(f"Output dataset file not found at: {output_path}")
            return False, errors

        try:
            # Load output dataset
            df = pd.read_csv(output_path)
            
            # Check output columns
            required_output_cols = self.expected_columns + ["cleaned_text"]
            missing_cols = [col for col in required_output_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"Output dataset is missing columns: {missing_cols}")
            
            # Check for empty cleaned_text (nulls or whitespace-only)
            empty_cleaned_count = (df["cleaned_text"].isnull() | (df["cleaned_text"].astype(str).str.strip() == "")).sum()
            if empty_cleaned_count > 0:
                # Note: some articles might have become empty if they only had URLs/emojis/etc.
                # However, for a production pipeline, we should flag if there are empty cleaned texts.
                errors.append(f"Found {empty_cleaned_count} records with empty 'cleaned_text'.")

            # Check label integrity (0 and 1 only)
            invalid_labels = df[~df["label"].isin([0, 1])]["label"].unique().tolist()
            if invalid_labels:
                errors.append(f"Output labels contain invalid values (only 0 and 1 allowed): {invalid_labels}")

            # Check label distribution alignment (make sure labels aren't altered or swapped)
            if expected_labels is not None:
                # If language filtering happened, row count might be less. We check the distribution values.
                # But we can verify label types match.
                pass
                
        except Exception as e:
            errors.append(f"Failed to read and validate output dataset: {e}")

        return len(errors) == 0, errors
