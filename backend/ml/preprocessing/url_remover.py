import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class URLRemover:
    """Removes URLs (http, https, www) from text."""
    
    def __init__(self) -> None:
        self.removed_count = 0
        # Regex pattern matching http://, https://, and www. URLs
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def transform(self, text: str) -> str:
        """Removes URLs from the text."""
        if not isinstance(text, str) or not text:
            return ""

        urls = self.url_pattern.findall(text)
        if urls:
            self.removed_count += len(urls)
            text = self.url_pattern.sub("", text)
            
        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
