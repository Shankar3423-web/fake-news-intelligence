import logging

logger = logging.getLogger("preprocessing_pipeline")

class LowercaseConverter:
    """Converts raw text to lowercase."""
    
    def __init__(self) -> None:
        pass
        
    def transform(self, text: str) -> str:
        """Converts string to lowercase."""
        if not isinstance(text, str):
            return ""
        return text.lower()
