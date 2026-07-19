import os
import yaml
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "steps": {
        "statistical_features": True,
        "readability_features": True,
        "lexical_features": True,
        "tfidf_features": True,
        "linguistic_features": True,
        "symbol_features": True,
    },
    "tfidf": {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
    },
    "short_words": {
        "max_length": 3,
    },
    "long_words": {
        "min_length": 6,
    },
    "stopwords": {
        "language": "english",
    },
    "linguistic": {
        "spacy_model": "en_core_web_sm",
    },
    "pipeline": {
        "pipeline_batch_size": 1000,
        "spacy_batch_size": 128,
        "batch_size": 1000,
    },
    "paths": {
        "input_dataset": "ml/preprocessing/processed/preprocessed_dataset_v1.csv",
        "output_dataset": "ml/feature_engineering/processed/feature_dataset_v1.csv",
        "tfidf_vectorizer": "ml/feature_engineering/processed/tfidf_vectorizer.joblib",
        "tfidf_matrix": "ml/feature_engineering/processed/tfidf_matrix.joblib",
        "processed_dir": "ml/feature_engineering/processed",
        "reports_dir": "ml/feature_engineering/reports",
        "statistics_dir": "ml/feature_engineering/statistics",
        "logs_dir": "ml/feature_engineering/logs",
        "versions_dir": "ml/feature_engineering/versions",
        "statistics_file": "ml/feature_engineering/statistics/feature_statistics.json",
        "profile_file": "ml/feature_engineering/statistics/feature_profile.json",
        "quality_report_file": "ml/feature_engineering/reports/feature_quality_report.md",
        "hash_file": "ml/feature_engineering/statistics/feature_hash.json",
        "versions_file": "ml/feature_engineering/statistics/feature_versions.json",
    }
}

class FeatureConfig:
    """
    Configuration manager for the Phase 4 Feature Engineering module.
    Loads and validates the configuration from YAML format.
    """
    def __init__(self, config_path: str = "config/feature_config.yaml") -> None:
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
        """Returns True if the specified feature engineering step is enabled."""
        return self._config.get("steps", {}).get(step_name, True)

    @property
    def steps(self) -> Dict[str, bool]:
        """Gets dictionary of step enablement toggles."""
        return self._config.get("steps", {})

    @property
    def tfidf_max_features(self) -> int:
        """Maximum features for TF-IDF."""
        return self._config.get("tfidf", {}).get("max_features", 5000)

    @property
    def tfidf_ngram_range(self) -> Tuple[int, int]:
        """Term n-gram range for TF-IDF."""
        val = self._config.get("tfidf", {}).get("ngram_range", [1, 2])
        return (val[0], val[1]) if isinstance(val, list) and len(val) == 2 else (1, 2)

    @property
    def tfidf_min_df(self) -> int:
        """Minimum document frequency for TF-IDF."""
        return self._config.get("tfidf", {}).get("min_df", 2)

    @property
    def tfidf_max_df(self) -> float:
        """Maximum document frequency for TF-IDF."""
        return self._config.get("tfidf", {}).get("max_df", 0.95)

    @property
    def tfidf_sublinear_tf(self) -> bool:
        """Apply sublinear scaling to term frequencies."""
        return self._config.get("tfidf", {}).get("sublinear_tf", True)

    @property
    def max_short_word_length(self) -> int:
        """Maximum length for a word to be considered short."""
        return self._config.get("short_words", {}).get("max_length", 3)

    @property
    def min_long_word_length(self) -> int:
        """Minimum length for a word to be considered long."""
        return self._config.get("long_words", {}).get("min_length", 6)

    @property
    def stopword_language(self) -> str:
        """Language name for NLTK stopword list."""
        return self._config.get("stopwords", {}).get("language", "english")

    @property
    def spacy_model(self) -> str:
        """spaCy model name to load."""
        return self._config.get("linguistic", {}).get("spacy_model", "en_core_web_sm")

    @property
    def batch_size(self) -> int:
        """Batch size for processing (backward compatibility)."""
        return self._config.get("pipeline", {}).get("pipeline_batch_size", 
               self._config.get("pipeline", {}).get("batch_size", 1000))

    @property
    def pipeline_batch_size(self) -> int:
        """Pipeline batch size for outer chunks."""
        return self._config.get("pipeline", {}).get("pipeline_batch_size", 1000)

    @property
    def spacy_batch_size(self) -> int:
        """Internal batch size for spaCy nlp.pipe."""
        return self._config.get("pipeline", {}).get("spacy_batch_size", 128)

    def get_path(self, path_key: str) -> str:
        """Resolves relative file or directory paths."""
        return self._config.get("paths", {}).get(path_key, "")
