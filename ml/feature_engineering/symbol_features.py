import pandas as pd
import string
import logging
from typing import Dict, Any

logger = logging.getLogger("feature_engineering_pipeline")

class SymbolFeatureExtractor:
    """
    Extracts symbol and orthographic features from raw text.
    Features:
    - Digit Count
    - Uppercase Character Count
    - Punctuation Count
    - Special Character Count (non-alphanumeric, non-whitespace, non-punctuation)
    """
    def __init__(self) -> None:
        self.punctuation_set = set(string.punctuation)

    def extract_features(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extracts symbol features from a series of raw texts.
        """
        logger.info("Extracting symbol features...")
        features = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                features.append({
                    "sym_digit_count": 0,
                    "sym_uppercase_count": 0,
                    "sym_punctuation_count": 0,
                    "sym_special_char_count": 0
                })
                continue
                
            try:
                digit_count = sum(1 for c in text if c.isdigit())
                uppercase_count = sum(1 for c in text if c.isupper())
                punctuation_count = sum(1 for c in text if c in self.punctuation_set)
                
                # Special characters: anything not alphanumeric, whitespace, or punctuation
                special_char_count = sum(
                    1 for c in text 
                    if not c.isalnum() and not c.isspace() and c not in self.punctuation_set
                )
                
                features.append({
                    "sym_digit_count": digit_count,
                    "sym_uppercase_count": uppercase_count,
                    "sym_punctuation_count": punctuation_count,
                    "sym_special_char_count": special_char_count
                })
            except Exception as e:
                logger.error(f"Error extracting symbol features for text: {e}")
                features.append({
                    "sym_digit_count": 0,
                    "sym_uppercase_count": 0,
                    "sym_punctuation_count": 0,
                    "sym_special_char_count": 0
                })
                
        return pd.DataFrame(features)
