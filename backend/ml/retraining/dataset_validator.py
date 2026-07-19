"""
dataset_validator.py
Validates training and merged datasets for Phase 12.
"""
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger("retraining_pipeline")

# Expected column types for the feature dataset (label must be 0/1 integer)
_LABEL_VALID_VALUES = {0, 1}


class DatasetValidator:
    """
    Validates dataset integrity before and after merging approved feedback.

    Checks performed
    ----------------
    - Schema completeness (required columns present).
    - No duplicate record identifiers.
    - Valid label values (0 or 1).
    - Non-empty dataset.
    - No all-NaN feature columns (structural check).
    """

    def __init__(
        self,
        label_column: str = "label",
        id_column: str = "id",
    ) -> None:
        self._label_col = label_column
        self._id_col = id_column

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def validate(
        self,
        df: pd.DataFrame,
        context: str = "dataset",
        required_columns: List[str] | None = None,
        check_duplicates: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Validates a DataFrame and returns ``(is_valid, errors)``.

        Args:
            df:               DataFrame to validate.
            context:          Human-readable label used in error messages.
            required_columns: Additional required column names beyond
                              *label_column*.
            check_duplicates: Whether to check for duplicate IDs.

        Returns:
            Tuple of ``(True, [])`` when valid, else ``(False, [error_messages])``.
        """
        errors: List[str] = []

        if df is None or df.empty:
            errors.append(f"{context}: Dataset is empty.")
            return False, errors

        # 1. Required column presence
        required = list(required_columns or [])
        if self._label_col not in required:
            required.append(self._label_col)

        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            errors.append(
                f"{context}: Missing required columns: {missing_cols}."
            )

        # 2. Label validity
        if self._label_col in df.columns:
            invalid_labels = set(df[self._label_col].dropna().unique()) - _LABEL_VALID_VALUES
            if invalid_labels:
                errors.append(
                    f"{context}: Invalid label values detected: {invalid_labels}. "
                    f"Expected: {_LABEL_VALID_VALUES}."
                )

            null_labels = int(df[self._label_col].isna().sum())
            if null_labels > 0:
                errors.append(
                    f"{context}: {null_labels} records have null labels."
                )

        # 3. Duplicate identifier check
        if check_duplicates and self._id_col in df.columns:
            dup_count = int(df[self._id_col].duplicated().sum())
            if dup_count > 0:
                errors.append(
                    f"{context}: {dup_count} duplicate '{self._id_col}' values found."
                )

        # 4. All-NaN columns (structural integrity)
        all_nan_cols = [
            c
            for c in df.columns
            if c not in (self._label_col, self._id_col)
            and df[c].isna().all()
        ]
        if all_nan_cols:
            errors.append(
                f"{context}: Columns with all-NaN values: {all_nan_cols[:10]} "
                f"(showing first 10)."
            )

        is_valid = len(errors) == 0
        if is_valid:
            logger.info(
                "%s: Validation passed (%d rows, %d columns).",
                context,
                len(df),
                len(df.columns),
            )
        else:
            for err in errors:
                logger.error("Validation error – %s", err)

        return is_valid, errors

    def validate_schema_match(
        self,
        reference_df: pd.DataFrame,
        candidate_df: pd.DataFrame,
        context: str = "schema-match",
    ) -> Tuple[bool, List[str]]:
        """
        Checks that *candidate_df* has all columns that *reference_df* has.

        Args:
            reference_df: The reference (existing training) dataset.
            candidate_df: The merged/new candidate dataset.
            context:      Human-readable label for error messages.

        Returns:
            Tuple ``(True, [])`` when schemas match, else ``(False, errors)``.
        """
        errors: List[str] = []
        ref_cols = set(reference_df.columns)
        cand_cols = set(candidate_df.columns)

        missing_in_candidate = ref_cols - cand_cols
        if missing_in_candidate:
            errors.append(
                f"{context}: Candidate is missing columns from reference: "
                f"{sorted(missing_in_candidate)}."
            )

        if errors:
            logger.error("Schema mismatch detected: %s", errors)
        else:
            logger.info("%s: Schema validation passed.", context)

        return len(errors) == 0, errors
