import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class WhitespaceNormalizer:
    """Normalizes multiple spaces, tabs, and newlines into a single space and strips padding."""
    
    def __init__(self) -> None:
        self.pattern = re.compile(r'\s+')

    def transform(self, text: str) -> str:
        """Applies whitespace normalization to the text."""
        if not isinstance(text, str):
            return ""
        return self.pattern.sub(" ", text).strip()
