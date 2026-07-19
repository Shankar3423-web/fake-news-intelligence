import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class SpecialCharacterRemover:
    """Removes non-alphanumeric special characters (symbols, control characters) from text."""
    
    def __init__(self) -> None:
        self.removed_count = 0
        # Matches characters that are NOT: alphanumeric, whitespace, or common punctuation
        self.pattern = re.compile(r"[^a-zA-Z0-9\s.,!?;:\"'()\[\]{}@#$%\^&*_\-+=<>?/\\|~`]")

    def transform(self, text: str) -> str:
        """Removes special characters from the text."""
        if not isinstance(text, str) or not text:
            return ""

        matches = self.pattern.findall(text)
        if matches:
            self.removed_count += len(matches)
            text = self.pattern.sub("", text)
            
        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
