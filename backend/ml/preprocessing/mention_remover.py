import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class MentionRemover:
    """Removes @mentions (e.g. @username) from text."""
    
    def __init__(self) -> None:
        self.removed_count = 0
        # Matches @ followed by word characters (letters, numbers, underscores)
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')

    def transform(self, text: str) -> str:
        """Removes @mentions from the text."""
        if not isinstance(text, str) or not text:
            return ""

        mentions = self.mention_pattern.findall(text)
        if mentions:
            self.removed_count += len(mentions)
            text = self.mention_pattern.sub("", text)
            
        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
