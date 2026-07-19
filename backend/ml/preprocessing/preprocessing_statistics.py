import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger("preprocessing_pipeline")

class PreprocessingStatistics:
    """
    Tracks, accumulates, and saves preprocessing run metrics.
    Uses running sums to ensure memory efficiency on large datasets.
    """
    
    def __init__(self) -> None:
        self.rows_processed = 0
        self.rows_failed = 0
        self.languages: Dict[str, int] = {}
        
        self.total_tokens = 0
        self.total_words = 0
        self.total_characters = 0
        self.total_clean_length = 0
        
        self.stopwords_removed = 0
        self.urls_removed = 0
        self.emails_removed = 0
        self.html_tags_removed = 0
        self.emoji_count = 0
        self.special_character_count = 0
        
        self.execution_time = 0.0

    def update_row_metrics(
        self, 
        original_text: str, 
        clean_text: str, 
        tokens_count: int
    ) -> None:
        """Accumulates metrics for a successfully processed row."""
        self.rows_processed += 1
        
        # Original text metrics
        if isinstance(original_text, str):
            self.total_characters += len(original_text)
            self.total_words += len(original_text.split())
            
        # Clean text metrics
        if isinstance(clean_text, str):
            self.total_clean_length += len(clean_text)
            
        self.total_tokens += tokens_count

    def update_cleaner_stats(self, stats: Dict[str, int]) -> None:
        """Merges counters from the text cleaner class."""
        self.html_tags_removed += stats.get("html_tags_removed", 0)
        self.urls_removed += stats.get("urls_removed", 0)
        self.emails_removed += stats.get("emails_removed", 0)
        self.emoji_count += stats.get("emojis_removed", 0)
        self.special_character_count += stats.get("special_characters_removed", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Compiles all statistics into a dictionary with correct JSON keys."""
        avg_token_count = (self.total_tokens / self.rows_processed) if self.rows_processed > 0 else 0.0
        avg_word_count = (self.total_words / self.rows_processed) if self.rows_processed > 0 else 0.0
        avg_character_count = (self.total_characters / self.rows_processed) if self.rows_processed > 0 else 0.0
        avg_clean_text_len = (self.total_clean_length / self.rows_processed) if self.rows_processed > 0 else 0.0

        return {
            "Rows Processed": self.rows_processed,
            "Rows Failed": self.rows_failed,
            "Languages": self.languages,
            "Average Token Count": round(avg_token_count, 2),
            "Average Word Count": round(avg_word_count, 2),
            "Average Character Count": round(avg_character_count, 2),
            "Average Clean Text Length": round(avg_clean_text_len, 2),
            "Stopwords Removed": self.stopwords_removed,
            "URLs Removed": self.urls_removed,
            "Emails Removed": self.emails_removed,
            "HTML Tags Removed": self.html_tags_removed,
            "Emoji Count": self.emoji_count,
            "Special Character Count": self.special_character_count,
            "Execution Time": round(self.execution_time, 2)
        }

    def save(self, file_path: str) -> None:
        """Saves compiled statistics to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Statistics successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save statistics to {file_path}: {e}")
