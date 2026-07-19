import os
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger("admin_review_pipeline")

DEFAULT_CONFIG: Dict[str, Any] = {
    "admin_review": {
        "allowed_review_states": ["APPROVED", "REJECTED", "PENDING"],
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
        "output_dir": "ml/admin_review",
        "logs_dir": "ml/admin_review/logs",
        "reports_dir": "ml/admin_review/reports",
        "statistics_dir": "ml/admin_review/statistics",
        "metadata_dir": "ml/admin_review/metadata",
        "history_dir": "ml/admin_review/history",
        "approved_dir": "ml/admin_review/approved",
        "hashes_dir": "ml/admin_review/hashes",
        "versions_dir": "ml/admin_review/versions",
        "charts_dir": "ml/admin_review/charts",
        
        "feedback_history_csv": "ml/feedback/history/feedback_history.csv",
        "admin_review_history_csv": "ml/admin_review/history/admin_review_history.csv",
        "approved_feedback_csv": "ml/admin_review/approved/approved_feedback.csv",
        "admin_review_statistics_json": "ml/admin_review/statistics/admin_review_statistics.json",
        "admin_review_metadata_json": "ml/admin_review/metadata/admin_review_metadata.json",
        "admin_review_report_md": "ml/admin_review/reports/admin_review_report.md",
        "admin_review_hashes_json": "ml/admin_review/hashes/admin_review_hashes.json",
        "admin_review_versions_json": "ml/admin_review/versions/admin_review_versions.json",
        "admin_review_pipeline_log": "ml/admin_review/logs/admin_review_pipeline.log"
    }
}

class AdminReviewConfig:
    """
    Configuration manager for the Phase 11 Admin Review System.
    """
    def __init__(self, config_path: str = "ml/admin_review/admin_review_config.yaml") -> None:
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
    def allowed_review_states(self) -> List[str]:
        return list(self._config.get("admin_review", {}).get("allowed_review_states", ["APPROVED", "REJECTED", "PENDING"]))

    @property
    def system_version(self) -> str:
        return str(self._config.get("admin_review", {}).get("system_version", "1.0.0"))

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
