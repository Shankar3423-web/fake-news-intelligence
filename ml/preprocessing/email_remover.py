import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class EmailRemover:
    """Removes email addresses from text."""
    
    def __init__(self) -> None:
        self.removed_count = 0
        # Regex matching typical email patterns
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    def transform(self, text: str) -> str:
        """Removes email addresses from the text."""
        if not isinstance(text, str) or not text:
            return ""

        emails = self.email_pattern.findall(text)
        if emails:
            self.removed_count += len(emails)
            text = self.email_pattern.sub("", text)
            
        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
