import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger("prediction_pipeline")

DEFAULT_CONFIG: Dict[str, Any] = {
    "text_length": {
        "min_length": 10,
        "max_length": 50000
    },
    "logging": {
        "level": "INFO",
        "log_file": "ml/prediction/logs/prediction_pipeline.log"
    },
    "exports": {
        "enable_prediction_export": True,
        "enable_statistics": True,
        "enable_reports": True,
        "enable_metadata": True,
        "enable_hashing": True,
        "enable_versions": True,
        "enable_charts": True
    },
    "paths": {
        "best_model_json": "ml/evaluation/best_model.json",
        "models_dir": "ml/training/models",
        "selected_features_json": "ml/feature_selection/processed/selected_feature_names.json",
        "tfidf_vectorizer": "ml/feature_engineering/processed/tfidf_vectorizer.joblib",
        "output_dir": "ml/prediction",
        "logs_dir": "ml/prediction/logs",
        "reports_dir": "ml/prediction/reports",
        "statistics_dir": "ml/prediction/statistics",
        "metadata_dir": "ml/prediction/metadata",
        "hashes_dir": "ml/prediction/hashes",
        "versions_dir": "ml/prediction/versions",
        "predictions_dir": "ml/prediction/predictions",
        "charts_dir": "ml/prediction/charts",
        "prediction_history_csv": "ml/prediction/predictions/prediction_history.csv",
        "prediction_statistics_json": "ml/prediction/statistics/prediction_statistics.json",
        "prediction_report_md": "ml/prediction/reports/prediction_report.md",
        "prediction_metadata_json": "ml/prediction/metadata/prediction_metadata.json",
        "prediction_hashes_json": "ml/prediction/hashes/prediction_hashes.json",
        "prediction_versions_json": "ml/prediction/versions/prediction_versions.json"
    }
}

class PredictionConfig:
    """
    Configuration manager for the Phase 8 Prediction (Inference) Engine.
    Loads and validates the configuration from YAML format.
    """
    def __init__(self, config_path: str = "ml/prediction/prediction_config.yaml") -> None:
        self.config_path = config_path
        self._config = self._load_config()
        self._create_directories()

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

    def _create_directories(self) -> None:
        """Ensures all configured output directories exist."""
        for path_key, path_val in self._config.get("paths", {}).items():
            if path_key.endswith("_dir") and isinstance(path_val, str):
                os.makedirs(path_val, exist_ok=True)

    @property
    def min_text_length(self) -> int:
        return int(self._config.get("text_length", {}).get("min_length", 10))

    @property
    def max_text_length(self) -> int:
        return int(self._config.get("text_length", {}).get("max_length", 50000))

    @property
    def logging_level(self) -> str:
        return str(self._config.get("logging", {}).get("level", "INFO"))

    @property
    def log_file(self) -> str:
        return str(self._config.get("logging", {}).get("log_file", "ml/prediction/logs/prediction_pipeline.log"))

    @property
    def enable_prediction_export(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_prediction_export", True))

    @property
    def enable_statistics(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_statistics", True))

    @property
    def enable_reports(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_reports", True))

    @property
    def enable_metadata(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_metadata", True))

    @property
    def enable_hashing(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_hashing", True))

    @property
    def enable_versions(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_versions", True))

    @property
    def enable_charts(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_charts", True))

    def get_path(self, path_key: str) -> str:
        return str(self._config.get("paths", {}).get(path_key, ""))
