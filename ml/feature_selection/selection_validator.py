import os
import logging
import pandas as pd
from typing import Tuple, List, Optional
from ml.feature_selection.selection_config import SelectionConfig

logger = logging.getLogger("feature_selection_pipeline")

class SelectionValidator:
    """
    Validates structural integrity, sample alignment, and correct schema
    for the input and output datasets of the feature selection module.
    """
    def __init__(self) -> None:
        self.base_columns = ["id", "label", "cleaned_text"]

    def validate_inputs(self, config: SelectionConfig) -> Tuple[bool, List[str]]:
        """
        Validates that all required input files exist.
        """
        errors = []
        input_csv = config.get_path("input_csv")
        input_tfidf_matrix = config.get_path("input_tfidf_matrix")
        input_tfidf_vectorizer = config.get_path("input_tfidf_vectorizer")

        if not os.path.exists(input_csv):
            errors.append(f"Input dense dataset missing: {input_csv}")
        if not os.path.exists(input_tfidf_matrix):
            errors.append(f"Input TF-IDF sparse matrix missing: {input_tfidf_matrix}")
        if not os.path.exists(input_tfidf_vectorizer):
            errors.append(f"Input TF-IDF vectorizer missing: {input_tfidf_vectorizer}")

        # Check CSV header if it exists
        if os.path.exists(input_csv):
            try:
                df_head = pd.read_csv(input_csv, nrows=5)
                for col in ["id", "label", "cleaned_text"]:
                    if col not in df_head.columns:
                        errors.append(f"Input CSV is missing required column: {col}")
            except Exception as e:
                errors.append(f"Failed to parse input CSV header: {e}")

        return len(errors) == 0, errors

    def validate_outputs(
        self, 
        input_csv_path: str,
        output_csv_path: str,
        expected_features: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validates output dataset structure and integrity.
        Verifies:
        - Output CSV file exists.
        - Sample count matches the input CSV.
        - Sample order is preserved (checking 'id' row-by-row).
        - Labels are preserved (checking 'label' row-by-row).
        - Base columns ('id', 'label', 'cleaned_text') exist in output.
        - All expected selected feature columns exist in output.
        - No null/NaN values exist in the output feature columns.
        """
        errors = []
        
        if not os.path.exists(output_csv_path):
            errors.append(f"Output CSV dataset does not exist: {output_csv_path}")
            return False, errors

        try:
            logger.info("Reading input and output datasets for cross-validation...")
            # Load only ID and label from input to save memory
            df_in = pd.read_csv(input_csv_path, usecols=["id", "label"])
            df_out = pd.read_csv(output_csv_path)

            # 1. Validate row counts
            if len(df_in) != len(df_out):
                errors.append(f"Row count mismatch! Input has {len(df_in)} rows, output has {len(df_out)} rows.")
                return False, errors  # Cannot do item-level comparisons if lengths differ

            # 2. Verify sample order preservation (checking IDs)
            if not df_in["id"].equals(df_out["id"]):
                errors.append("Sample order mismatch! Output IDs do not match input IDs row-for-row.")

            # 3. Verify label preservation
            if not df_in["label"].equals(df_out["label"]):
                errors.append("Label mismatch! Output labels do not match input labels row-for-row.")

            # 4. Verify base columns presence
            for col in self.base_columns:
                if col not in df_out.columns:
                    errors.append(f"Output CSV is missing base column: {col}")

            # 5. Verify feature columns presence
            missing_feats = [feat for feat in expected_features if feat not in df_out.columns]
            if missing_feats:
                errors.append(f"Output CSV is missing expected selected features: {missing_feats}")

            # 6. Verify null counts in feature columns
            present_feats = [feat for feat in expected_features if feat in df_out.columns]
            if present_feats:
                null_counts = df_out[present_feats].isnull().sum()
                cols_with_nulls = null_counts[null_counts > 0]
                if not cols_with_nulls.empty:
                    for col, count in cols_with_nulls.items():
                        errors.append(f"Feature column '{col}' contains {count} null/NaN values.")

            # 7. Check that output CSV has no extra columns other than base and selected features
            expected_cols_set = set(self.base_columns + expected_features)
            extra_cols = [col for col in df_out.columns if col not in expected_cols_set]
            if extra_cols:
                # Warning rather than error, or could be error. Let's make it a warning.
                logger.warning(f"Output CSV contains extra columns not in the selection set: {extra_cols}")

        except Exception as e:
            errors.append(f"Validation failed with unexpected exception: {e}")

        return len(errors) == 0, errors
