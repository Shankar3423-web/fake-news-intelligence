import os
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "steps": {
        "lowercase_conversion": True,
        "unicode_normalization": True,
        "remove_html_tags": True,
        "remove_urls": True,
        "remove_emails": True,
        "remove_mentions": True,
        "process_hashtags": True,
        "remove_emojis": True,
        "remove_special_characters": True,
        "handle_punctuation": True,
        "handle_numbers": True,
        "whitespace_normalization": True,
        "expand_contractions": True,
        "language_detection": True,
        "tokenization": True,
        "stopword_removal": True,
        "lemmatization": True,
        "short_word_remover": True,
    },
    "unicode": {
        "normalization_form": "NFKC",
    },
    "numbers": {
        "replacement_token": "<NUM>",
        "preserve_alphanumeric": True,
    },
    "stopwords": {
        "language": "english",
        "custom_stopwords": [],
    },
    "lemmatizer": {
        "spacy_model": "en_core_web_sm",
    },
    "short_words": {
        "min_length": 2,
    },
    "language_detection": {
        "supported_languages": ["en"],
        "default_language": "en",
        "fallback_on_error": True,
    },
    "pipeline": {
        "batch_size": 5000,
    },
    "paths": {
        "input_dataset": "ml/dataset/processed/master_dataset_v1.csv",
        "output_dataset": "ml/preprocessing/processed/preprocessed_dataset_v1.csv",
        "processed_dir": "ml/preprocessing/processed",
        "reports_dir": "ml/preprocessing/reports",
        "statistics_dir": "ml/preprocessing/statistics",
        "logs_dir": "ml/preprocessing/logs",
        "versions_dir": "ml/preprocessing/versions",
        "statistics_file": "ml/preprocessing/statistics/preprocessing_statistics.json",
        "profile_file": "ml/preprocessing/statistics/preprocessing_profile.json",
        "quality_report_file": "ml/preprocessing/reports/preprocessing_quality_report.md",
        "versions_file": "ml/preprocessing/statistics/preprocessing_versions.json",
        "hash_file": "ml/preprocessing/statistics/preprocessing_hash.json",
    }
}

class PreprocessingConfig:
    """
    Configuration manager for the NLP preprocessing module.
    Loads and validates the configuration from YAML format.
    """
    def __init__(self, config_path: str = "config/preprocessing_config.yaml") -> None:
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from YAML file with fallback to default config."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found at {self.config_path}. Using default configuration.")
            return DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if not loaded:
                    logger.warning("Empty configuration file. Using default configuration.")
                    return DEFAULT_CONFIG.copy()
                
                # Merge loaded configuration with defaults to ensure all keys exist
                return self._merge_dicts(DEFAULT_CONFIG, loaded)
        except Exception as e:
            logger.error(f"Error loading configuration from {self.config_path}: {e}. Using defaults.")
            return DEFAULT_CONFIG.copy()

    def _merge_dicts(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merges a loaded dictionary into a default dictionary."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def get_step_enabled(self, step_name: str) -> bool:
        """Returns True if the specified preprocessing step is enabled."""
        return self._config.get("steps", {}).get(step_name, True)

    @property
    def steps(self) -> Dict[str, bool]:
        """Gets dictionary of step enablement toggles."""
        return self._config.get("steps", {})

    @property
    def unicode_form(self) -> str:
        """Unicode normalization form (NFC, NFKC, NFD, NFKD)."""
        return self._config.get("unicode", {}).get("normalization_form", "NFKC")

    @property
    def number_replacement_token(self) -> str:
        """Replacement token for standalone numbers."""
        return self._config.get("numbers", {}).get("replacement_token", "<NUM>")

    @property
    def preserve_alphanumeric_numbers(self) -> bool:
        """True if alphanumeric numbers like COVID19 should be preserved."""
        return self._config.get("numbers", {}).get("preserve_alphanumeric", True)

    @property
    def stopword_language(self) -> str:
        """Language name for NLTK stopword list."""
        return self._config.get("stopwords", {}).get("language", "english")

    @property
    def custom_stopwords(self) -> List[str]:
        """Custom user stopwords list."""
        return self._config.get("stopwords", {}).get("custom_stopwords", [])

    @property
    def spacy_model(self) -> str:
        """spaCy model name to load."""
        return self._config.get("lemmatizer", {}).get("spacy_model", "en_core_web_sm")

    @property
    def min_token_length(self) -> int:
        """Minimum length for tokens to keep."""
        return self._config.get("short_words", {}).get("min_length", 2)

    @property
    def supported_languages(self) -> List[str]:
        """List of supported languages for the text."""
        return self._config.get("language_detection", {}).get("supported_languages", ["en"])

    @property
    def default_language(self) -> str:
        """Default language to fall back on detection failure."""
        return self._config.get("language_detection", {}).get("default_language", "en")

    @property
    def fallback_on_error(self) -> bool:
        """True to fallback to default language if language detection raises an error."""
        return self._config.get("language_detection", {}).get("fallback_on_error", True)

    @property
    def batch_size(self) -> int:
        """Batch size for processing dataframes."""
        return self._config.get("pipeline", {}).get("batch_size", 5000)

    def get_path(self, path_key: str) -> str:
        """Returns relative path configured for the given path key."""
        return self._config.get("paths", {}).get(path_key, DEFAULT_CONFIG["paths"][path_key])
