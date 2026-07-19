import pandas as pd
import numpy as np
import logging
import textstat
from typing import Dict, Any, List

logger = logging.getLogger("feature_engineering_pipeline")

class StatisticalFeatureExtractor:
    """
    Extracts statistical features from the text.
    Features:
    - Word Count
    - Character Count
    - Sentence Count
    - Average Word Length
    - Average Sentence Length
    - Vocabulary Size
    """
    def __init__(self) -> None:
        pass

    def extract_features(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extracts statistical features from a series of raw texts.
        """
        logger.info("Extracting statistical features...")
        features = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                # Default values for empty or null text
                features.append({
                    "stat_word_count": 0,
                    "stat_char_count": 0,
                    "stat_sentence_count": 0,
                    "stat_avg_word_length": 0.0,
                    "stat_avg_sentence_length": 0.0,
                    "stat_vocabulary_size": 0
                })
                continue
            
            try:
                # Basic tokenization by splitting on whitespace
                words = [w for w in text.split() if w]
                word_count = len(words)
                char_count = len(text)
                
                # Use textstat for sentence count
                try:
                    sentence_count = textstat.sentence_count(text)
                except Exception:
                    # Fallback to simple split if textstat fails
                    sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
                
                # If sentence count is zero, default to 1 to avoid ZeroDivisionError
                sentence_count = max(1, sentence_count)
                
                # Average word length
                if word_count > 0:
                    avg_word_len = sum(len(w) for w in words) / word_count
                else:
                    avg_word_len = 0.0
                
                # Average sentence length
                avg_sentence_len = word_count / sentence_count
                
                # Vocabulary size (unique lowercase words)
                vocab_size = len(set(w.lower() for w in words))
                
                features.append({
                    "stat_word_count": word_count,
                    "stat_char_count": char_count,
                    "stat_sentence_count": sentence_count,
                    "stat_avg_word_length": float(round(avg_word_len, 4)),
                    "stat_avg_sentence_length": float(round(avg_sentence_len, 4)),
                    "stat_vocabulary_size": vocab_size
                })
            except Exception as e:
                logger.error(f"Error extracting statistical features for text: {e}")
                features.append({
                    "stat_word_count": 0,
                    "stat_char_count": 0,
                    "stat_sentence_count": 0,
                    "stat_avg_word_length": 0.0,
                    "stat_avg_sentence_length": 0.0,
                    "stat_vocabulary_size": 0
                })
                
        return pd.DataFrame(features)
