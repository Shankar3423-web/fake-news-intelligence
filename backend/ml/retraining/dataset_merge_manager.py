"""
dataset_merge_manager.py
Merges approved feedback records with the existing training dataset for Phase 12.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.retraining.retraining_config import RetrainingConfig

logger = logging.getLogger("retraining_pipeline")


class DatasetMergeManager:
    """
    Merges administrator-approved feedback with the existing training dataset.

    Strategy
    --------
    Phase 12 works at the **feature-selected dataset level** (Phase 5 output).
    Approved feedback records carry only a ``label`` integer derived from the
    admin decision; they do NOT carry the full 283-dimensional feature vector.

    To produce a merged training dataset that is compatible with the existing
    feature space, the merge manager:

    1.  Loads the existing feature-selected CSV (contains label + feature cols).
    2.  For each approved feedback record, checks if its ``Feedback ID``
        already exists in the training set (by ``id`` column) and either
        updates or appends accordingly.
    3.  When no matching ID exists, the approved record is **not** silently
        ignored—instead a synthetic row with the feedback label is appended
        with zero-filled features so training can proceed with the extra signal.
        This is the correct behaviour when the feedback is about a new article
        not in the original corpus.
    4.  Deduplicates by the configured strategy (first / last).
    5.  Optionally shuffles the merged dataset.
    """

    def __init__(self, config: RetrainingConfig) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────────────────────────
    def merge(
        self,
        existing_df: pd.DataFrame,
        approved_df: pd.DataFrame,
        feature_columns: List[str],
    ) -> Tuple[bool, str, pd.DataFrame]:
        """
        Merges *approved_df* into *existing_df*.

        Args:
            existing_df:     Existing training DataFrame (Phase 5 output).
            approved_df:     Approved feedback DataFrame (from loader).
            feature_columns: List of feature column names (Phase 5 features).

        Returns:
            Tuple ``(success, error_message, merged_dataframe)``.
        """
        label_col = self._config.get("training_dataset", "label_column") or "label"
        id_col = self._config.get("training_dataset", "id_column") or "id"
        duplicate_strategy = self._config.merge_duplicate_strategy

        logger.info(
            "Starting dataset merge: existing=%d rows, approved=%d records.",
            len(existing_df),
            len(approved_df),
        )

        try:
            # Build approved contribution rows
            new_rows: List[Dict[str, Any]] = []
            for _, row in approved_df.iterrows():
                record: Dict[str, Any] = {}

                # Feedback ID as surrogate ID
                fid = str(row.get("Feedback ID", "")).strip()
                record[id_col] = fid if fid else f"fb_{len(new_rows)}"

                # Label from mapped integer
                record[label_col] = int(row["label"])

                # Zero-fill all feature columns (unknown features for new article)
                for feat in feature_columns:
                    record[feat] = 0.0

                new_rows.append(record)

            new_df = pd.DataFrame(new_rows)

            # Ensure column alignment before concatenation
            for col in existing_df.columns:
                if col not in new_df.columns:
                    new_df[col] = 0.0

            new_df = new_df[existing_df.columns]

            # Concatenate
            merged = pd.concat([existing_df, new_df], ignore_index=True)

            # Deduplicate by ID column
            if id_col in merged.columns:
                keep = "first" if duplicate_strategy == "first" else "last"
                before = len(merged)
                merged = merged.drop_duplicates(subset=[id_col], keep=keep)
                after = len(merged)
                if before != after:
                    logger.info(
                        "Deduplicated %d rows using strategy='%s'.",
                        before - after,
                        duplicate_strategy,
                    )

            # Shuffle
            if self._config.merge_shuffle:
                merged = merged.sample(
                    frac=1,
                    random_state=self._config.random_state,
                ).reset_index(drop=True)

            logger.info(
                "Merge complete: %d → %d rows (+%d from feedback).",
                len(existing_df),
                len(merged),
                max(0, len(merged) - len(existing_df)),
            )
            return True, "", merged

        except Exception as exc:  # noqa: BLE001
            msg = f"Dataset merge failed: {exc}"
            logger.exception(msg)
            return False, msg, pd.DataFrame()

    # ── Statistics ────────────────────────────────────────────────────────────────────────────────
    def compute_merge_stats(
        self,
        existing_df: pd.DataFrame,
        approved_df: pd.DataFrame,
        merged_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Returns a statistics dictionary describing the merge operation.

        Args:
            existing_df: Original training dataset.
            approved_df: Approved feedback records.
            merged_df:   Post-merge dataset.

        Returns:
            Dictionary with counts, deltas, and label distributions.
        """
        label_col = self._config.get("training_dataset", "label_column") or "label"

        def _label_dist(df: pd.DataFrame) -> Dict[str, int]:
            if label_col not in df.columns:
                return {}
            return {str(k): int(v) for k, v in df[label_col].value_counts().to_dict().items()}

        return {
            "existing_rows": len(existing_df),
            "approved_records": len(approved_df),
            "merged_rows": len(merged_df),
            "added_rows": max(0, len(merged_df) - len(existing_df)),
            "label_distribution_existing": _label_dist(existing_df),
            "label_distribution_approved": _label_dist(approved_df),
            "label_distribution_merged": _label_dist(merged_df),
        }
