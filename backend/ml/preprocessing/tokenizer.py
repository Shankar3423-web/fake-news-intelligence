import logging
from typing import List
from nltk.tokenize import word_tokenize

logger = logging.getLogger("preprocessing_pipeline")

class Tokenizer:
    """Tokenizes string text into a list of word tokens using NLTK."""
    
    def __init__(self) -> None:
        pass

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into words."""
        if not isinstance(text, str) or not text:
            return []
        
        try:
            return word_tokenize(text)
        except Exception as e:
            logger.debug(f"NLTK word_tokenize failed: {e}. Falling back to whitespace splitting.")
            return text.split()
