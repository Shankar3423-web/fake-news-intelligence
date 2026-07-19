import logging
from typing import Dict, List
import langdetect
from langdetect import DetectorFactory

# Set seed for langdetect to be deterministic
DetectorFactory.seed = 0

logger = logging.getLogger("preprocessing_pipeline")

class LanguageDetector:
    """
    Detects the language of articles using langdetect.
    Maintains statistics on detected languages.
    """
    
    def __init__(self, supported_languages: List[str] = None, default_language: str = "en", fallback_on_error: bool = True) -> None:
        self.supported_languages = supported_languages or ["en"]
        self.default_language = default_language
        self.fallback_on_error = fallback_on_error
        self.language_counts: Dict[str, int] = {}
        self.rows_processed = 0
        self.rows_failed = 0

    def detect(self, text: str) -> str:
        """
        Detects the language of the text.
        If detection fails, returns default_language if fallback_on_error is True, else raises.
        """
        if not isinstance(text, str) or not text.strip():
            self.language_counts[self.default_language] = self.language_counts.get(self.default_language, 0) + 1
            return self.default_language

        self.rows_processed += 1
        try:
            # langdetect can sometimes be slow on very long text; we can truncate to speed up
            # usually first 500 characters are more than enough to detect language accurately
            sample_text = text[:1000]
            lang = langdetect.detect(sample_text)
            self.language_counts[lang] = self.language_counts.get(lang, 0) + 1
            return lang
        except Exception as e:
            self.rows_failed += 1
            logger.debug(f"Language detection failed for text (len={len(text)}): {e}")
            if self.fallback_on_error:
                self.language_counts[self.default_language] = self.language_counts.get(self.default_language, 0) + 1
                return self.default_language
            else:
                raise e

    def is_supported(self, lang: str) -> bool:
        """Checks if the language code is in the supported list."""
        return lang in self.supported_languages

    def get_report(self) -> Dict[str, int]:
        """Returns the distribution of detected languages."""
        return self.language_counts.copy()

    def reset(self) -> None:
        """Resets the statistics."""
        self.language_counts = {}
        self.rows_processed = 0
        self.rows_failed = 0
