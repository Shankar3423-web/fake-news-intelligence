import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger("model_training_pipeline")

class DatasetLoader:
    """
    Loads the selected feature dataset and performs comprehensive validations:
    - Verifies file existence of both dataset CSV and feature list JSON.
    - Validates schema (base columns + feature columns).
    - Checks for duplicate column names.
    - Validates numeric type for feature columns.
    - Checks for missing values (NaN/Null) in features and label.
    - Validates that label column is binary (0/1).
    - Validates class distribution to ensure both classes are represented.
    """
    def __init__(self, dataset_path: str, feature_names_path: str) -> None:
        self.dataset_path = dataset_path
        self.feature_names_path = feature_names_path
        self.base_columns = ["id", "label", "cleaned_text"]

    def load_and_validate(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Loads the selected feature dataset and performs all validations.
        
        Returns:
            Tuple of:
                df: Loaded pandas DataFrame
                selected_features: List of feature column names
                
        Raises:
            FileNotFoundError: If dataset or feature names file does not exist.
            ValueError: If schema, labels, types, duplicate columns, missing values,
                        or label distributions are invalid.
        """
        # 1. File existence validation
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Selected feature dataset file not found at: {self.dataset_path}")
        if not os.path.exists(self.feature_names_path):
            raise FileNotFoundError(f"Selected feature names JSON file not found at: {self.feature_names_path}")

        # 2. Load feature names
        logger.info(f"Loading selected feature names from {self.feature_names_path}...")
        try:
            with open(self.feature_names_path, "r", encoding="utf-8") as f:
                selected_features = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse feature names JSON: {e}")

        if not isinstance(selected_features, list) or len(selected_features) == 0:
            raise ValueError("Feature names JSON must contain a non-empty list of strings.")

        # 3. Load dataset
        logger.info(f"Loading dataset from {self.dataset_path}...")
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to read dataset CSV: {e}")

        # 4. Validate duplicate columns in file
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            raise ValueError(f"Dataset contains duplicate column names: {set(duplicates)}")

        # 5. Validate schema (base columns)
        for col in self.base_columns:
            if col not in df.columns:
                raise ValueError(f"Required base column '{col}' is missing from the dataset.")

        # 6. Validate feature count and names existence
        missing_features = [feat for feat in selected_features if feat not in df.columns]
        if missing_features:
            raise ValueError(f"Dataset is missing {len(missing_features)} expected features. Missing: {missing_features[:10]}...")

        # 7. Validate numeric feature types
        non_numeric_features = []
        for feat in selected_features:
            if not np.issubdtype(df[feat].dtype, np.number):
                non_numeric_features.append(feat)
        if non_numeric_features:
            raise ValueError(f"Features must be of numeric type. Non-numeric features found: {non_numeric_features[:10]}")

        # 8. Validate missing values (NaN/Null) in features and label
        null_counts = df[selected_features].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise ValueError(f"Missing values found in feature columns: {cols_with_nulls.to_dict()}")

        if df["label"].isnull().any():
            raise ValueError("Label column contains missing values (NaN/Null).")

        # 9. Validate labels (must be binary 0 or 1)
        unique_labels = df["label"].unique().tolist()
        invalid_labels = [label for label in unique_labels if label not in [0, 1]]
        if invalid_labels:
            raise ValueError(f"Label column contains invalid values. Only [0, 1] allowed. Found: {unique_labels}")

        # 10. Validate label distribution
        label_counts = df["label"].value_counts().to_dict()
        if 0 not in label_counts or 1 not in label_counts:
            raise ValueError(f"Dataset is imbalanced or missing one class. Label counts: {label_counts}")
        
        # Check if any class has fewer than 5 samples (sanity check)
        for cls, count in label_counts.items():
            if count < 5:
                raise ValueError(f"Class {cls} has insufficient samples: {count}")

        logger.info(f"Dataset loaded and validated successfully.")
        logger.info(f"Total samples: {len(df)}, Feature count: {len(selected_features)}")
        logger.info(f"Class distribution: {label_counts}")

        return df, selected_features
