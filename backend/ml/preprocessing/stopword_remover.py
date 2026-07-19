import logging
from typing import List, Set
from nltk.corpus import stopwords

logger = logging.getLogger("preprocessing_pipeline")

class StopwordRemover:
    """Removes stopwords from a list of tokens using NLTK stopwords and optional custom additions."""
    
    def __init__(self, language: str = "english", custom_stopwords: List[str] = None) -> None:
        self.language = language
        self.removed_count = 0
        
        try:
            self.stopwords: Set[str] = set(stopwords.words(self.language))
        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}. Using a fallback basic set.")
            self.stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "on", "for", "with", "by"}
            
        if custom_stopwords:
            self.stopwords.update([w.lower() for w in custom_stopwords])

    def remove(self, tokens: List[str]) -> List[str]:
        """Removes stopwords from the token list and increments the removed counter."""
        if not tokens:
            return []

        filtered_tokens = []
        for token in tokens:
            if token.lower() in self.stopwords:
                self.removed_count += 1
            else:
                filtered_tokens.append(token)
                
        return filtered_tokens

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
