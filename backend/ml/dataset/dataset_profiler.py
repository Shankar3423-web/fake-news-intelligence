import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatasetProfiler:
    """
    Generates detailed, production-quality profiling analytics of a dataset.
    Calculates size metrics, class distributions, category and source breakdowns,
    missing values, article lengths, and saves the report as JSON.
    """

    def __init__(self) -> None:
        pass

    def profile_dataset(
        self, 
        df: pd.DataFrame, 
        duplicate_count: int = 0,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates the profiling report from the DataFrame.
        
        Args:
            df: The processed master DataFrame.
            duplicate_count: Number of duplicates identified during pipeline run.
            output_path: Path to save the JSON profile. If None, it is not saved to disk.
            
        Returns:
            A dictionary containing profiling analytics.
        """
        logger.info("Generating dataset profiling report.")
        total_articles = len(df)
        
        if total_articles == 0:
            logger.warning("Empty DataFrame passed to profiler.")
            return {"error": "Dataset is empty"}

        # Class counts
        real_articles = int((df["label"] == 0).sum())
        fake_articles = int((df["label"] == 1).sum())

        # Origin counts
        indian_articles = int((df["dataset_origin"] == "INDIA").sum())
        isot_articles = int((df["dataset_origin"] == "ISOT").sum())

        # Distributions (value counts converted to dict)
        lang_dist = df["language"].value_counts(dropna=False).to_dict()
        lang_dist = {str(k): int(v) for k, v in lang_dist.items()}
        
        cat_dist = df["category"].value_counts(dropna=False).to_dict()
        cat_dist = {str(k): int(v) for k, v in cat_dist.items()}

        source_dist = df["source"].value_counts(dropna=False).to_dict()
        source_dist = {str(k): int(v) for k, v in source_dist.items()}

        # Missing values (None / NaN) count and percentage per column
        missing_values = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = float(round((null_count / total_articles) * 100, 4))
            missing_values[col] = {
                "count": null_count,
                "percentage": null_pct
            }

        # Duplicate calculation
        # If duplicate_count is passed, we compute percentage based on (final + duplicate_count) as total
        initial_total = total_articles + duplicate_count
        duplicate_pct = float(round((duplicate_count / initial_total) * 100, 4)) if initial_total > 0 else 0.0

        # Article text lengths (in characters and words)
        text_series = df["text"].astype(str).str.strip()
        char_lengths = text_series.str.len()
        word_lengths = text_series.str.split().str.len()

        avg_char_len = float(round(char_lengths.mean(), 2))
        avg_word_len = float(round(word_lengths.mean(), 2))

        # Longest article
        max_char_idx = char_lengths.idxmax()
        longest_article = {
            "id": str(df.loc[max_char_idx, "id"]) if "id" in df.columns else str(max_char_idx),
            "character_length": int(char_lengths.max()),
            "word_length": int(word_lengths.loc[max_char_idx]),
            "title": str(df.loc[max_char_idx, "title"])[:100]
        }

        # Shortest article
        min_char_idx = char_lengths.idxmin()
        shortest_article = {
            "id": str(df.loc[min_char_idx, "id"]) if "id" in df.columns else str(min_char_idx),
            "character_length": int(char_lengths.min()),
            "word_length": int(word_lengths.loc[min_char_idx]),
            "title": str(df.loc[min_char_idx, "title"])[:100]
        }

        # Top Categories and Sources
        top_categories = df["category"].dropna().value_counts().head(5).to_dict()
        top_categories = {str(k): int(v) for k, v in top_categories.items()}

        top_sources = df["source"].dropna().value_counts().head(5).to_dict()
        top_sources = {str(k): int(v) for k, v in top_sources.items()}

        # Build Profile Report Dict
        report = {
            "summary": {
                "total_articles": total_articles,
                "real_articles": real_articles,
                "fake_articles": fake_articles,
                "indian_articles": indian_articles,
                "isot_articles": isot_articles,
                "duplicate_count": duplicate_count,
                "duplicate_percentage": duplicate_pct
            },
            "distributions": {
                "language": lang_dist,
                "category": cat_dist,
                "source": source_dist
            },
            "top_analytics": {
                "top_categories": top_categories,
                "top_sources": top_sources
            },
            "text_statistics": {
                "average_character_length": avg_char_len,
                "average_word_length": avg_word_len,
                "longest_article": longest_article,
                "shortest_article": shortest_article
            },
            "missing_values": missing_values
        }

        # Save to file if output_path is provided
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=4, ensure_ascii=False)
                logger.info(f"Saved profiling report to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save profiling report to {output_path}. Error: {str(e)}")

        return report
