import os
import logging
import pandas as pd
from typing import Tuple, List, Optional

logger = logging.getLogger("feature_engineering_pipeline")

class FeatureValidator:
    """
    Validates input and output datasets for structural integrity,
    schema adherence, and value correctness.
    """
    def __init__(self) -> None:
        self.required_input_columns = ["id", "label", "text", "cleaned_text"]
        
        # All dense features that must exist in output
        self.expected_feature_columns = [
            # Statistical
            "stat_word_count", "stat_char_count", "stat_sentence_count",
            "stat_avg_word_length", "stat_avg_sentence_length", "stat_vocabulary_size",
            # Readability
            "read_flesch_reading_ease", "read_flesch_kincaid_grade",
            "read_smog", "read_gunning_fog", "read_coleman_liau",
            # Lexical
            "lex_diversity", "lex_unique_words", "lex_stopword_ratio",
            "lex_long_word_ratio", "lex_short_word_ratio",
            # Linguistic
            "ling_entity_count", "ling_noun_count", "ling_verb_count", "ling_adj_count",
            "ling_pos_noun_ratio", "ling_pos_verb_ratio", "ling_pos_adj_ratio",
            "ling_pos_adv_ratio", "ling_pos_pron_ratio",
            # Symbols
            "sym_digit_count", "sym_uppercase_count", 
            "sym_punctuation_count", "sym_special_char_count"
        ]

    def validate_input(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validates that the input preprocessed dataset exists and has the required schema.
        """
        errors = []
        if not os.path.exists(file_path):
            errors.append(f"Input dataset file not found at: {file_path}")
            return False, errors

        try:
            # Read the header only to minimize memory usage
            df_head = pd.read_csv(file_path, nrows=5)
            
            missing_cols = [col for col in self.required_input_columns if col not in df_head.columns]
            if missing_cols:
                errors.append(f"Input dataset is missing required columns: {missing_cols}")
                
        except Exception as e:
            errors.append(f"Failed to read input dataset header: {e}")

        return len(errors) == 0, errors

    def validate_output(
        self, 
        output_path: str, 
        expected_rows: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validates the output feature dataset.
        Checks:
        - Output file exists
        - Required output columns (id, label, cleaned_text, and all dense features) exist
        - No null/NaN values exist in any engineered feature column
        - Row count matches expected_rows (if provided)
        - Labels only contain 0 and 1
        """
        errors = []
        if not os.path.exists(output_path):
            errors.append(f"Output dataset file not found at: {output_path}")
            return False, errors

        try:
            # Load the output dataset
            df = pd.read_csv(output_path)
            
            # 1. Check required base columns
            base_cols = ["id", "label", "cleaned_text"]
            missing_base = [col for col in base_cols if col not in df.columns]
            if missing_base:
                errors.append(f"Output dataset is missing base columns: {missing_base}")
                
            # 2. Check required engineered feature columns
            missing_features = [col for col in self.expected_feature_columns if col not in df.columns]
            if missing_features:
                errors.append(f"Output dataset is missing engineered feature columns: {missing_features}")
                
            # 3. Check for Null/NaN values in the feature columns
            feature_cols_present = [col for col in self.expected_feature_columns if col in df.columns]
            null_counts = df[feature_cols_present].isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]
            if not cols_with_nulls.empty:
                for col, count in cols_with_nulls.items():
                    errors.append(f"Feature column '{col}' contains {count} null/NaN values.")
                    
            # 4. Check row count consistency
            if expected_rows is not None and len(df) != expected_rows:
                errors.append(f"Row count mismatch. Expected {expected_rows} rows, but output has {len(df)}.")
                
            # 5. Check labels
            invalid_labels = df[~df["label"].isin([0, 1])]["label"].unique().tolist()
            if invalid_labels:
                errors.append(f"Labels contain invalid values (only 0 and 1 allowed): {invalid_labels}")
                
        except Exception as e:
            errors.append(f"Failed to read and validate output dataset: {e}")

        return len(errors) == 0, errors
