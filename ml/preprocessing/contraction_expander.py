import logging
import contractions

logger = logging.getLogger("preprocessing_pipeline")

class ContractionExpander:
    """Expands English contractions (e.g., don't -> do not) using the contractions library."""
    
    def __init__(self) -> None:
        pass

    def transform(self, text: str) -> str:
        """Expands contractions in the text."""
        if not isinstance(text, str) or not text:
            return ""
        
        try:
            return contractions.fix(text)
        except Exception as e:
            logger.debug(f"Contraction expansion failed: {e}. Returning original text.")
            return text
