import pandas as pd
import string
import logging
from nltk.corpus import stopwords
from typing import Dict, Any, Set
from ml.feature_engineering.feature_config import FeatureConfig

logger = logging.getLogger("feature_engineering_pipeline")

class LexicalFeatureExtractor:
    """
    Extracts lexical features from text.
    Features:
    - Lexical Diversity
    - Unique Words
    - Stopword Ratio
    - Long Word Ratio
    - Short Word Ratio
    """
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.stopword_lang = config.stopword_language
        self.max_short_len = config.max_short_word_length
        self.min_long_len = config.min_long_word_length
        self.stopwords_set = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        """Loads NLTK stopwords list for the configured language."""
        try:
            return set(stopwords.words(self.stopword_lang))
        except Exception as e:
            logger.error(f"Failed to load NLTK stopwords for '{self.stopword_lang}': {e}. Using empty set.")
            return set()

    def extract_features(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extracts lexical features from a series of raw texts.
        """
        logger.info("Extracting lexical features...")
        features = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                features.append({
                    "lex_diversity": 0.0,
                    "lex_unique_words": 0,
                    "lex_stopword_ratio": 0.0,
                    "lex_long_word_ratio": 0.0,
                    "lex_short_word_ratio": 0.0
                })
                continue
                
            try:
                # Tokenize and clean words by stripping punctuation
                words = []
                for w in text.split():
                    w_clean = w.strip(string.punctuation).lower()
                    if w_clean:
                        words.append(w_clean)
                
                word_count = len(words)
                
                if word_count == 0:
                    features.append({
                        "lex_diversity": 0.0,
                        "lex_unique_words": 0,
                        "lex_stopword_ratio": 0.0,
                        "lex_long_word_ratio": 0.0,
                        "lex_short_word_ratio": 0.0
                    })
                    continue
                
                unique_words = set(words)
                unique_count = len(unique_words)
                
                # Lexical Diversity
                diversity = unique_count / word_count
                
                # Stopword Count
                stopword_count = sum(1 for w in words if w in self.stopwords_set)
                stopword_ratio = stopword_count / word_count
                
                # Long Words (length >= min_long_len)
                long_count = sum(1 for w in words if len(w) >= self.min_long_len)
                long_ratio = long_count / word_count
                
                # Short Words (length <= max_short_len)
                short_count = sum(1 for w in words if len(w) <= self.max_short_len)
                short_ratio = short_count / word_count
                
                features.append({
                    "lex_diversity": float(round(diversity, 4)),
                    "lex_unique_words": unique_count,
                    "lex_stopword_ratio": float(round(stopword_ratio, 4)),
                    "lex_long_word_ratio": float(round(long_ratio, 4)),
                    "lex_short_word_ratio": float(round(short_ratio, 4))
                })
            except Exception as e:
                logger.error(f"Error extracting lexical features for text: {e}")
                features.append({
                    "lex_diversity": 0.0,
                    "lex_unique_words": 0,
                    "lex_stopword_ratio": 0.0,
                    "lex_long_word_ratio": 0.0,
                    "lex_short_word_ratio": 0.0
                })
                
        return pd.DataFrame(features)
