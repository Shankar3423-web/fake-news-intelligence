import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Dynamic column mapping to identify raw columns and convert to standardized schema
COLUMN_MAPPING = {
    # title mappings
    "title": "title",
    "Title": "title",
    "headline": "title",
    "Headline": "title",
    "statement": "title",
    "Statement": "title",
    "heading": "title",
    "Heading": "title",
    
    # text mappings
    "text": "text",
    "Text": "text",
    "content": "text",
    "Content": "text",
    "body": "text",
    "Body": "text",
    "article": "text",
    "Article": "text",
    "news": "text",
    "News": "text",
    
    # label mappings
    "label": "label",
    "Label": "label",
    "class": "label",
    "Class": "label",
    "target": "label",
    "Target": "label",
    
    # source mappings
    "source": "source",
    "Source": "source",
    "publisher": "source",
    "Publisher": "source",
    
    # category mappings
    "category": "category",
    "Category": "category",
    "subject": "category",
    "Subject": "category",
    "topic": "category",
    "Topic": "category",
    
    # author mappings
    "author": "author",
    "Author": "author",
    "writer": "author",
    "Writer": "author",
    
    # published_date mappings
    "published_date": "published_date",
    "date": "published_date",
    "Date": "published_date",
    "time": "published_date",
    "Time": "published_date",
    "published_at": "published_date",
    
    # language mappings
    "language": "language",
    "Language": "language",
    "lang": "language",
    
    # url mappings
    "url": "url",
    "URL": "url",
    "link": "url",
    "Link": "url",
}

class DatasetStandardizer:
    """
    Standardizes a DataFrame into the master schema.
    Trims strings, normalizes dates, maps labels to binary (0/1),
    adds origin tracking, and generates unique IDs.
    """
    
    TARGET_COLUMNS = [
        "id", "title", "text", "label", "source", "category", 
        "author", "published_date", "language", "url", "dataset_origin"
    ]
    
    def __init__(self) -> None:
        pass

    def standardize(
        self, 
        df: pd.DataFrame, 
        origin: str, 
        default_label: Optional[int] = None,
        default_lang: str = "en"
    ) -> pd.DataFrame:
        """
        Transforms the raw DataFrame to standard schema.
        
        Args:
            df: Raw input DataFrame.
            origin: Origin identifier ('ISOT' or 'INDIA').
            default_label: Fallback label if not found in raw dataset.
            default_lang: Fallback language code.
            
        Returns:
            Standardized DataFrame with final schema.
        """
        logger.info(f"Standardizing dataset with origin: {origin}, shape: {df.shape}")
        
        # 1. Create a copy to prevent modifying original
        df_std = df.copy()
        
        # 2. Map columns dynamically
        column_rename = {}
        for col in df_std.columns:
            if col in COLUMN_MAPPING:
                column_rename[col] = COLUMN_MAPPING[col]
        
        df_std = df_std.rename(columns=column_rename)
        logger.debug(f"Renamed columns: {column_rename}")
        
        # 3. Add default columns if missing in standardized DataFrame
        for target_col in self.TARGET_COLUMNS:
            if target_col not in df_std.columns and target_col != "id":
                df_std[target_col] = None

        # 4. Handle Labels (Real = 0, Fake = 1)
        if "label" not in df_std.columns or df_std["label"].isnull().all():
            if default_label is not None:
                df_std["label"] = default_label
                logger.info(f"Assigned default label {default_label} to all rows.")
            else:
                df_std["label"] = 0 # Default to real if none exists
                logger.warning("No label column found and no default label provided. Defaulting to 0.")
        else:
            # Map raw labels
            df_std["label"] = df_std["label"].apply(self._normalize_single_label, default_val=default_label)

        # 5. Set Dataset Origin
        df_std["dataset_origin"] = origin.upper()

        # 6. Normalize Dates
        if "published_date" in df_std.columns:
            df_std["published_date"] = self._normalize_dates(df_std["published_date"])

        # 7. Normalize Language
        if "language" in df_std.columns:
            df_std["language"] = df_std["language"].apply(self._normalize_language, default_lang=default_lang)
        else:
            df_std["language"] = default_lang

        # 8. Trim Whitespace for all string columns
        for col in df_std.columns:
            # Only trim string columns
            if df_std[col].dtype == object or isinstance(df_std[col].dtype, pd.StringDtype):
                df_std[col] = df_std[col].astype(str).str.strip()
                # Replace empty string with None/NaN
                df_std[col] = df_std[col].replace("", None)
                # Replace representation of 'nan' or 'None' as string with None
                df_std[col] = df_std[col].replace({"nan": None, "None": None, "NaN": None, "<NA>": None})

        # 9. Clean/fill optional values with None (specifically float NaNs to None)
        # Use object type to support Python None
        df_std = df_std.where(pd.notnull(df_std), None)

        # 10. Reorder columns to target schema (excluding id for now, we will add id next)
        cols_to_keep = [col for col in self.TARGET_COLUMNS if col != "id"]
        df_std = df_std[cols_to_keep]

        # 11. Generate Unique IDs (format: {origin.lower()}_{index})
        # Reset index to guarantee consecutive index matching the length
        df_std = df_std.reset_index(drop=True)
        df_std.insert(0, "id", df_std.index.map(lambda idx: f"{origin.lower()}_{idx + 1}"))

        logger.info(f"Standardization complete. Final columns: {df_std.columns.tolist()}, shape: {df_std.shape}")
        return df_std

    def _normalize_single_label(self, val: Any, default_val: Optional[int] = None) -> int:
        """
        Normalizes a single label value to binary 0 (Real) or 1 (Fake).
        """
        if pd.isna(val) or val is None:
            if default_val is not None:
                return default_val
            return 0
            
        val_str = str(val).strip().lower()
        
        # Real representations
        if val_str in ["0", "0.0", "real", "true", "real news", "true news", "true.csv"]:
            return 0
        # Fake representations
        elif val_str in ["1", "1.0", "fake", "false", "fake news", "false news", "fake.csv"]:
            return 1
        else:
            # Fallback
            if default_val is not None:
                return default_val
            # Try parsing numeric value
            try:
                numeric_val = float(val)
                return 1 if numeric_val >= 0.5 else 0
            except ValueError:
                return 1 # Default to fake if unparseable

    def _normalize_dates(self, date_series: pd.Series) -> pd.Series:
        """
        Attempts to parse and normalize date strings to ISO YYYY-MM-DD format.
        """
        def parse_single_date(d: Any) -> Optional[str]:
            if pd.isna(d) or d is None:
                return None
            
            d_str = str(d).strip()
            if not d_str or d_str.lower() in ["nan", "none", "null"]:
                return None

            # Handle common cleaning on string date formats (e.g. remove text like 'Published:', 'on ', etc.)
            d_clean = re.sub(r"^(published|on|at|date)\s*:\s*", "", d_str, flags=re.IGNORECASE).strip()
            
            # Try general pandas datetime parser
            try:
                # errors='raise' to test validity
                dt = pd.to_datetime(d_clean, errors="raise")
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

            # Try parsing text-based formats manually or regex if pandas failed (e.g. 'December 31, 2017')
            try:
                dt = pd.to_datetime(d_clean, format="%B %d, %Y", errors="ignore")
                if not isinstance(dt, str):
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

            return d_clean # Return raw cleaned string if parsing fails, but logged as warning later

        return date_series.apply(parse_single_date)

    def _normalize_language(self, val: Any, default_lang: str = "en") -> str:
        """
        Normalizes language strings to standard ISO-639-1 two-letter codes.
        """
        if pd.isna(val) or val is None:
            return default_lang
            
        val_str = str(val).strip().lower()
        if val_str in ["english", "eng", "en_us", "en_gb", "en"]:
            return "en"
        elif val_str in ["hindi", "hin", "hi"]:
            return "hi"
        else:
            return val_str[:2] # Fallback to first 2 letters
