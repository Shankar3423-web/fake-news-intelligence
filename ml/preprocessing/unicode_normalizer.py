import unicodedata
import logging

logger = logging.getLogger("preprocessing_pipeline")

class UnicodeNormalizer:
    """Normalizes unicode representations in text (e.g. NFKC, NFC)."""
    
    def __init__(self, form: str = "NFKC") -> None:
        self.form = form

    def transform(self, text: str) -> str:
        """Applies unicode normalization form to the text."""
        if not isinstance(text, str):
            return ""
        return unicodedata.normalize(self.form, text)
