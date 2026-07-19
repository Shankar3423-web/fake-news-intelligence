import re
import logging

logger = logging.getLogger("preprocessing_pipeline")

class NumberHandler:
    """
    Handles numbers by replacing standalone numbers with a token (e.g., <NUM>),
    while preserving alphanumeric terms (e.g., COVID19, 3D, F15).
    """
    
    def __init__(self, replacement_token: str = "<NUM>", preserve_alphanumeric: bool = True) -> None:
        self.replacement_token = replacement_token
        self.preserve_alphanumeric = preserve_alphanumeric
        self.replaced_count = 0
        
        if preserve_alphanumeric:
            # Matches digits not preceded or followed by alphanumeric characters (word characters)
            # e.g., "123", "1,234.56", but NOT "COVID19" or "3D"
            # (?<!\w) checks that the character before is not alphanumeric/underscore
            # (?!\w) checks that the character after is not alphanumeric/underscore
            self.pattern = re.compile(r'(?<!\w)\d+(?:[.,]\d+)*(?!\w)')
        else:
            # Matches any sequence of digits, including parts of alphanumeric words
            self.pattern = re.compile(r'\d+(?:[.,]\d+)*')

    def transform(self, text: str) -> str:
        """Replaces standalone numbers with the token."""
        if not isinstance(text, str) or not text:
            return ""

        matches = self.pattern.findall(text)
        if matches:
            self.replaced_count += len(matches)
            text = self.pattern.sub(self.replacement_token, text)
            
        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.replaced_count = 0
