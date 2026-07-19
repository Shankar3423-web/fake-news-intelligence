import os
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DatasetValidator:
    """
    Validates raw datasets against specified rules and constraints.
    Generates structured, detailed validation reports.
    """

    def __init__(self) -> None:
        pass

    def validate_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validates file existence and extension support.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        if not os.path.exists(file_path):
            errors.append(f"File does not exist: {file_path}")
            return False, errors

        _, ext = os.path.splitext(file_path.lower())
        supported_exts = {".csv", ".xlsx", ".xls"}
        if ext not in supported_exts:
            errors.append(f"Unsupported file extension '{ext}'. Supported: {supported_exts}")

        return len(errors) == 0, errors

    def validate_dataframe(
        self, 
        df: pd.DataFrame, 
        file_label: str,
        schema_type: str
    ) -> Dict[str, Any]:
        """
        Performs in-depth validation on the standardized DataFrame.
        
        Args:
            df: The standardized pandas DataFrame.
            file_label: A string identifier for logging/reporting.
            schema_type: The schema type to validate against ("ISOT" or "INDIA").
            
        Returns:
            A validation report dictionary.
        """
        import numpy as np
        report: Dict[str, Any] = {
            "dataset_label": file_label,
            "schema_type": schema_type,
            "is_valid": True,
            "shape": list(df.shape),
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

        # 1. Check Empty Dataset
        if df.empty:
            report["is_valid"] = False
            report["errors"].append("Dataset is empty (contains 0 rows).")
            return report

        if len(df.columns) == 0:
            report["is_valid"] = False
            report["errors"].append("Dataset has 0 columns.")
            return report

        # 2. Check Duplicate Columns
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            report["is_valid"] = False
            report["errors"].append(f"Duplicate column names found: {duplicated_cols}")

        # 3. Check Schema Specific Columns presence in DataFrame
        common_columns = [
            "id", "title", "text", "label", "source", "category", 
            "author", "published_date", "language", "url", "dataset_origin"
        ]
        
        missing_common = [col for col in common_columns if col not in df.columns]
        if missing_common:
            report["is_valid"] = False
            report["errors"].append(f"Standardized DataFrame is missing expected target columns: {missing_common}")
            return report

        # Define which columns are expected to have values based on schema type
        if schema_type.upper() == "ISOT":
            # For ISOT, we expect title, text, label, category, published_date to have values.
            # source, author, language, url are optional and can be fully null.
            expected_fields = ["title", "text", "label", "category", "published_date"]
        elif schema_type.upper() == "INDIA":
            # For India, we expect id, title, text, label, source, category, author, published_date, language, url
            expected_fields = ["id", "title", "text", "label", "source", "category", "author", "published_date", "language", "url"]
        else:
            expected_fields = ["title", "text", "label"]

        # Check if expected fields exist and have at least some non-null values
        for field in expected_fields:
            null_count = int(df[field].isnull().sum())
            if null_count == len(df):
                if field in ["title", "text", "label"]:
                    report["is_valid"] = False
                    report["errors"].append(f"Mandatory schema column '{field}' is entirely null/empty.")
                else:
                    report["warnings"].append(f"Schema column '{field}' is entirely null/empty.")

        # 4. Empty Rows (Entire row is NaN/null in mandatory fields)
        mandatory_null_mask = df[["title", "text", "label"]].isnull().all(axis=1)
        empty_rows_count = int(mandatory_null_mask.sum())
        if empty_rows_count > 0:
            report["warnings"].append(f"Found {empty_rows_count} rows with completely empty mandatory fields.")
            report["metrics"]["empty_rows"] = empty_rows_count

        # 5. Missing Values in Mandatory Columns
        for field in ["title", "text", "label"]:
            missing_count = int(df[field].isnull().sum())
            if missing_count > 0:
                report["warnings"].append(f"Found {missing_count} missing values in mandatory column '{field}'.")
                report["metrics"][f"missing_{field}"] = missing_count

        # 6. Invalid Labels
        label_series = df["label"].dropna()
        unique_labels = label_series.unique().tolist()
        # Convert values to standard type for comparison
        clean_labels = []
        for val in unique_labels:
            try:
                clean_labels.append(int(float(val)))
            except (ValueError, TypeError):
                clean_labels.append(val)
                
        invalid_labels = [lbl for lbl in clean_labels if lbl not in [0, 1]]
        if invalid_labels:
            report["is_valid"] = False
            report["errors"].append(
                f"Label column contains invalid values: {invalid_labels}. Only 0 (Real) and 1 (Fake) are allowed."
            )
        report["metrics"]["labels"] = [int(x) for x in clean_labels if isinstance(x, (int, np.integer))]

        # 7. Corrupted Rows (check extremely short texts)
        text_series = df["text"].astype(str)
        short_rows_count = int((text_series.str.strip().str.len() < 3).sum())
        if short_rows_count > 0:
            report["warnings"].append(
                f"Found {short_rows_count} rows with text length less than 3 characters."
            )
            report["metrics"]["suspicious_short_text_rows"] = short_rows_count

        logger.info(f"Validation report for {file_label} ({schema_type}): is_valid={report['is_valid']}, errors={len(report['errors'])}, warnings={len(report['warnings'])}")
        return report
