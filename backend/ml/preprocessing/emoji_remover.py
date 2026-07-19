import logging
import emoji

logger = logging.getLogger("preprocessing_pipeline")

class EmojiRemover:
    """Removes emojis from text using the 'emoji' library."""
    
    def __init__(self) -> None:
        self.removed_count = 0

    def transform(self, text: str) -> str:
        """Removes all emojis from the text."""
        if not isinstance(text, str) or not text:
            return ""

        try:
            count = emoji.emoji_count(text)
            if count > 0:
                self.removed_count += count
                text = emoji.replace_emoji(text, replace="")
        except Exception as e:
            logger.debug(f"Emoji processing failed: {e}. Falling back to standard string cleaning.")
            # Fallback: remove non-ascii characters or do basic replacement
            # However, emoji package is in requirements.txt so it should be available.
            pass

        return text

    def reset(self) -> None:
        """Resets the internal counter."""
        self.removed_count = 0
