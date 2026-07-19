import re
import string
import logging

logger = logging.getLogger("preprocessing_pipeline")

class PunctuationHandler:
    """
    Handles punctuation by replacing it with space to prevent words from merging.
    Preserves apostrophes (') to ensure contractions can be expanded in later steps.
    """
    
    def __init__(self, replace_with: str = " ") -> None:
        self.replace_with = replace_with
        # Create a regex to match punctuation except apostrophe
        # string.punctuation is: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        punc_chars = string.punctuation.replace("'", "")
        # Escape characters for regex
        escaped_punc = "".join([f"\\{c}" for c in punc_chars])
        self.pattern = re.compile(f"[{escaped_punc}]")

    def transform(self, text: str) -> str:
        """Replaces punctuation characters (excluding apostrophes) with the replacement token."""
        if not isinstance(text, str) or not text:
            return ""

        text = self.pattern.sub(self.replace_with, text)
        return text
