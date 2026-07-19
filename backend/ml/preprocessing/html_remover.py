import re
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger("preprocessing_pipeline")

class HTMLRemover:
    """Removes HTML tags from text and decodes HTML entities."""
    
    def __init__(self) -> None:
        self.removed_count = 0
        # Simple regex to identify HTML tags
        self.tag_pattern = re.compile(r'<[^>]+>')

    def transform(self, text: str) -> str:
        """Removes HTML tags from the text."""
        if not isinstance(text, str) or not text:
            return ""

        # Quick check to avoid invoking parser for clean text
        if "<" in text and ">" in text:
            # Count tags
            tags = self.tag_pattern.findall(text)
            self.removed_count += len(tags)
            
            try:
                # Use BeautifulSoup to decode entities and strip tags
                soup = BeautifulSoup(text, "html.parser")
                text = soup.get_text()
            except Exception as e:
                logger.debug(f"BeautifulSoup parsing failed: {e}. Falling back to regex.")
                text = self.tag_pattern.sub(" ", text)

        return text
    
    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
