import logging
from typing import Any

logger = logging.getLogger("prediction_pipeline")

class InputValidator:
    """
    Validates raw news article text before feeding it to the pipeline.
    """
    def __init__(self, min_length: int = 10, max_length: int = 50000) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, text: Any) -> bool:
        """
        Validates the input text.
        Returns True if valid, raises ValueError with specific details if invalid.
        """
        logger.info("Validating raw text input...")
        
        # 1. Null / None check
        if text is None:
            raise ValueError("Input text cannot be Null/None.")
            
        # 2. Type check
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text).__name__}.")

        # 3. Unicode validity / encoding check
        try:
            text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Input text contains invalid unicode characters/encoding: {e}")

        # 4. Empty / Whitespace-only check
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace-only.")

        # 5. Minimum text length check
        length = len(stripped)
        if length < self.min_length:
            raise ValueError(
                f"Input text is too short ({length} characters). "
                f"Minimum required length is {self.min_length} characters."
            )

        # 6. Maximum text length check
        if length > self.max_length:
            raise ValueError(
                f"Input text is too long ({length} characters). "
                f"Maximum allowed length is {self.max_length} characters."
            )

        logger.info("Input text successfully validated.")
        return True
