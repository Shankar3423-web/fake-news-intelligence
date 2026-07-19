import logging
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class MissingValueHandler:
    """
    Handles missing values in datasets.
    Drops rows if mandatory fields (title, text, label) are missing or blank.
    Fills missing values in optional fields with None (representing SQL NULL).
    """

    MANDATORY_FIELDS = ["title", "text", "label"]

    def __init__(self) -> None:
        pass

    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Cleans missing values in the DataFrame.
        
        Args:
            df: Standardized input DataFrame.
            
        Returns:
            A tuple of (cleaned DataFrame, cleanup report dict).
        """
        initial_rows = len(df)
        logger.info(f"Handling missing values. Initial rows: {initial_rows}")

        report: Dict[str, Any] = {
            "initial_rows": initial_rows,
            "dropped_missing_title": 0,
            "dropped_missing_text": 0,
            "dropped_missing_label": 0,
            "total_dropped": 0,
            "final_rows": 0
        }

        if df.empty:
            report["final_rows"] = 0
            return df, report

        # Create a working copy
        df_clean = df.copy()

        # Identify missing mandatory fields
        # Note: A field is missing if it is null/NaN or if it is empty string/whitespace (for text/title)
        
        # 1. Missing Label
        label_missing_mask = df_clean["label"].isnull()
        report["dropped_missing_label"] = int(label_missing_mask.sum())
        
        # 2. Missing Title (null or whitespace only)
        title_missing_mask = df_clean["title"].isnull() | (df_clean["title"].astype(str).str.strip() == "")
        # But we only count if it wasn't already marked by missing label (to avoid double counting)
        report["dropped_missing_title"] = int((title_missing_mask & ~label_missing_mask).sum())
        
        # 3. Missing Text (null or whitespace only)
        text_missing_mask = df_clean["text"].isnull() | (df_clean["text"].astype(str).str.strip() == "")
        # Count if not already counted
        report["dropped_missing_text"] = int((text_missing_mask & ~label_missing_mask & ~title_missing_mask).sum())

        # Combine masks
        mandatory_missing_mask = label_missing_mask | title_missing_mask | text_missing_mask
        total_dropped = int(mandatory_missing_mask.sum())
        report["total_dropped"] = total_dropped
        report["removed_records"] = []

        # Extract dropped rows and annotate with reason
        if total_dropped > 0:
            dropped_df = df_clean[mandatory_missing_mask].copy()
            for idx, row in dropped_df.iterrows():
                reasons = []
                if label_missing_mask.loc[idx]:
                    reasons.append("Missing mandatory 'label' field")
                if title_missing_mask.loc[idx]:
                    reasons.append("Missing mandatory 'title' field")
                if text_missing_mask.loc[idx]:
                    reasons.append("Missing mandatory 'text' field")
                
                reason_str = ", ".join(reasons)
                report["removed_records"].append({
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "text": row.get("text"),
                    "label": row.get("label"),
                    "source": row.get("source"),
                    "category": row.get("category"),
                    "published_date": row.get("published_date"),
                    "dataset_origin": row.get("dataset_origin"),
                    "removal_reason": reason_str
                })

        # Drop the rows
        df_clean = df_clean[~mandatory_missing_mask].copy()
        
        # 4. Handle Optional Fields (fill with None)
        # Find all optional columns (non-mandatory)
        optional_cols = [col for col in df_clean.columns if col not in self.MANDATORY_FIELDS]
        for col in optional_cols:
            # Check for NaN / NaT / Null and fill with Python None
            df_clean[col] = df_clean[col].where(pd.notnull(df_clean[col]), None)
            
        final_rows = len(df_clean)
        report["final_rows"] = final_rows

        logger.info(
            f"Missing values handled. Dropped {total_dropped} rows due to missing mandatory fields. "
            f"Final rows: {final_rows}"
        )
        
        return df_clean, report
