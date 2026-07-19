import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class HashtagProcessor:
    """Processes hashtags by stripping the '#' but preserving the text."""
    
    def __init__(self) -> None:
        self.processed_count = 0
        # Matches # followed by word characters
        self.hashtag_pattern = re.compile(r'#([A-Za-z0-9_]+)')

    def transform(self, text: str) -> str:
        """Strips the '#' from hashtags in the text."""
        if not isinstance(text, str) or not text:
            return ""

        hashtags = self.hashtag_pattern.findall(text)
        if hashtags:
            self.processed_count += len(hashtags)
            text = self.hashtag_pattern.sub(r'\1', text)
            
        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.processed_count = 0
