import os
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger("feedback_pipeline")

DEFAULT_CONFIG: Dict[str, Any] = {
    "feedback": {
        "max_comment_length": 500,
        "min_comment_length": 3,
        "allowed_feedback_values": ["Correct", "Incorrect", "Unsure", "AGREE", "DISAGREE", "TRUE", "FALSE"],
        "allow_empty_comments": True,
        "system_version": "1.0.0"
    },
    "exports": {
        "enable_charts": True,
        "enable_reports": True,
        "enable_metadata": True,
        "enable_statistics": True,
        "enable_versioning": True,
        "enable_hash_generation": True
    },
    "paths": {
        "output_dir": "ml/feedback",
        "logs_dir": "ml/feedback/logs",
        "reports_dir": "ml/feedback/reports",
        "statistics_dir": "ml/feedback/statistics",
        "metadata_dir": "ml/feedback/metadata",
        "history_dir": "ml/feedback/history",
        "hashes_dir": "ml/feedback/hashes",
        "versions_dir": "ml/feedback/versions",
        "charts_dir": "ml/feedback/charts",
        "feedback_history_csv": "ml/feedback/history/feedback_history.csv",
        "feedback_statistics_json": "ml/feedback/statistics/feedback_statistics.json",
        "feedback_metadata_json": "ml/feedback/metadata/feedback_metadata.json",
        "feedback_report_md": "ml/feedback/reports/feedback_report.md",
        "feedback_hashes_json": "ml/feedback/hashes/feedback_hashes.json",
        "feedback_versions_json": "ml/feedback/versions/feedback_versions.json",
        "feedback_pipeline_log": "ml/feedback/logs/feedback_pipeline.log"
    }
}

class FeedbackConfig:
    """
    Configuration manager for the Phase 10 Feedback Collection System.
    """
    def __init__(self, config_path: str = "ml/feedback/feedback_config.yaml") -> None:
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
    def max_comment_length(self) -> int:
        return int(self._config.get("feedback", {}).get("max_comment_length", 500))

    @property
    def min_comment_length(self) -> int:
        return int(self._config.get("feedback", {}).get("min_comment_length", 3))

    @property
    def allowed_feedback_values(self) -> List[str]:
        return list(self._config.get("feedback", {}).get("allowed_feedback_values", ["Correct", "Incorrect", "Unsure", "AGREE", "DISAGREE", "TRUE", "FALSE"]))

    @property
    def allow_empty_comments(self) -> bool:
        return bool(self._config.get("feedback", {}).get("allow_empty_comments", True))

    @property
    def system_version(self) -> str:
        return str(self._config.get("feedback", {}).get("system_version", "1.0.0"))

    @property
    def enable_charts(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_charts", True))

    @property
    def enable_reports(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_reports", True))

    @property
    def enable_metadata(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_metadata", True))

    @property
    def enable_statistics(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_statistics", True))

    @property
    def enable_versioning(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_versioning", True))

    @property
    def enable_hash_generation(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_hash_generation", True))

    def get_path(self, path_key: str) -> str:
        return str(self._config.get("paths", {}).get(path_key, ""))
