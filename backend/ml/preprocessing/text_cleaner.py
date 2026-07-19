import logging
from typing import Dict, Any

from ml.preprocessing.preprocessing_config import PreprocessingConfig
from ml.preprocessing.lowercase_converter import LowercaseConverter
from ml.preprocessing.unicode_normalizer import UnicodeNormalizer
from ml.preprocessing.html_remover import HTMLRemover
from ml.preprocessing.url_remover import URLRemover
from ml.preprocessing.email_remover import EmailRemover
from ml.preprocessing.mention_remover import MentionRemover
from ml.preprocessing.hashtag_processor import HashtagProcessor
from ml.preprocessing.emoji_remover import EmojiRemover
from ml.preprocessing.special_character_remover import SpecialCharacterRemover
from ml.preprocessing.punctuation_handler import PunctuationHandler
from ml.preprocessing.number_handler import NumberHandler
from ml.preprocessing.whitespace_normalizer import WhitespaceNormalizer
from ml.preprocessing.contraction_expander import ContractionExpander

logger = logging.getLogger("preprocessing_pipeline")

class TextCleaner:
    """
    Coordinates and applies the string-level text cleaning steps (Steps 2 to 14)
    in the exact order defined by the preprocessing pipeline specification.
    """
    
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        
        # Instantiate cleaners
        self.lowercase = LowercaseConverter()
        self.unicode_norm = UnicodeNormalizer(form=config.unicode_form)
        self.html_remover = HTMLRemover()
        self.url_remover = URLRemover()
        self.email_remover = EmailRemover()
        self.mention_remover = MentionRemover()
        self.hashtag_processor = HashtagProcessor()
        self.emoji_remover = EmojiRemover()
        self.special_char_remover = SpecialCharacterRemover()
        self.punctuation_handler = PunctuationHandler()
        self.number_handler = NumberHandler(
            replacement_token=config.number_replacement_token,
            preserve_alphanumeric=config.preserve_alphanumeric_numbers
        )
        self.whitespace_norm = WhitespaceNormalizer()
        self.contraction_expander = ContractionExpander()

    def clean(self, text: str) -> str:
        """
        Applies enabled cleaning steps in the specified sequence.
        
        Sequence:
        2. Lowercase Conversion
        3. Unicode Normalization
        4. Remove HTML Tags
        5. Remove URLs
        6. Remove Email Addresses
        7. Remove @Mentions
        8. Process Hashtags
        9. Remove Emojis
        10. Remove Special Characters
        11. Handle Punctuation
        12. Handle Numbers
        13. Whitespace Normalization
        14. Expand English Contractions
        """
        if not isinstance(text, str):
            return ""

        # 2. Lowercase Conversion
        if self.config.get_step_enabled("lowercase_conversion"):
            text = self.lowercase.transform(text)

        # 3. Unicode Normalization
        if self.config.get_step_enabled("unicode_normalization"):
            text = self.unicode_norm.transform(text)

        # 4. Remove HTML Tags
        if self.config.get_step_enabled("remove_html_tags"):
            text = self.html_remover.transform(text)

        # 5. Remove URLs
        if self.config.get_step_enabled("remove_urls"):
            text = self.url_remover.transform(text)

        # 6. Remove Email Addresses
        if self.config.get_step_enabled("remove_emails"):
            text = self.email_remover.transform(text)

        # 7. Remove @Mentions
        if self.config.get_step_enabled("remove_mentions"):
            text = self.mention_remover.transform(text)

        # 8. Process Hashtags
        if self.config.get_step_enabled("process_hashtags"):
            text = self.hashtag_processor.transform(text)

        # 9. Remove Emojis
        if self.config.get_step_enabled("remove_emojis"):
            text = self.emoji_remover.transform(text)

        # 10. Remove Special Characters
        if self.config.get_step_enabled("remove_special_characters"):
            text = self.special_char_remover.transform(text)

        # 11. Handle Punctuation
        if self.config.get_step_enabled("handle_punctuation"):
            text = self.punctuation_handler.transform(text)

        # 12. Handle Numbers
        if self.config.get_step_enabled("handle_numbers"):
            text = self.number_handler.transform(text)

        # 13. Whitespace Normalization
        if self.config.get_step_enabled("whitespace_normalization"):
            text = self.whitespace_norm.transform(text)

        # 14. Expand English Contractions
        if self.config.get_step_enabled("expand_contractions"):
            text = self.contraction_expander.transform(text)
            
            # Since contraction expansion can theoretically introduce new whitespaces or double spaces,
            # we run a final quick whitespace normalization to ensure a clean single space formatting.
            text = self.whitespace_norm.transform(text)

        return text

    def get_stats(self) -> Dict[str, int]:
        """Collects metrics from individual cleaner components."""
        return {
            "html_tags_removed": self.html_remover.removed_count,
            "urls_removed": self.url_remover.removed_count,
            "emails_removed": self.email_remover.removed_count,
            "mentions_removed": self.mention_remover.removed_count,
            "hashtags_processed": self.hashtag_processor.processed_count,
            "emojis_removed": self.emoji_remover.removed_count,
            "special_characters_removed": self.special_char_remover.removed_count,
            "numbers_replaced": self.number_handler.replaced_count,
        }

    def reset(self) -> None:
        """Resets counters on all cleaners."""
        self.html_remover.reset()
        self.url_remover.reset()
        self.email_remover.reset()
        self.mention_remover.reset()
        self.hashtag_processor.reset()
        self.emoji_remover.reset()
        self.special_char_remover.reset()
        self.number_handler.reset()
