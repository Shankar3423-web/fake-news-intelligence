import logging
from typing import List

logger = logging.getLogger("preprocessing_pipeline")

class ShortWordRemover:
    """Removes tokens that are shorter than a configured minimum length."""
    
    def __init__(self, min_length: int = 2) -> None:
        self.min_length = min_length
        self.removed_count = 0

    def remove(self, tokens: List[str]) -> List[str]:
        """Removes short tokens from the list and tracks the count of removed tokens."""
        if not tokens:
            return []

        filtered_tokens = []
        for token in tokens:
            if len(token) < self.min_length:
                self.removed_count += 1
            else:
                filtered_tokens.append(token)
                
        return filtered_tokens

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
