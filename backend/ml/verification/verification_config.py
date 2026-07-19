import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger("verification_pipeline")

DEFAULT_CONFIG: Dict[str, Any] = {
    "verification": {
        "timeout": 10,
        "retry_count": 3,
        "rate_limit_delay": 1.0,
        "max_results_per_provider": 5,
        "similarity_threshold": 0.4,
        "evidence_threshold": 0.5,
        "duplicate_threshold": 0.85,
        "cache_expiration": 3600
    },
    "providers": {
        "enable_newsapi": True,
        "enable_gnews": True,
        "enable_newsdata": True
    },
    "exports": {
        "enable_charts": True,
        "enable_reports": True,
        "enable_metadata": True,
        "enable_statistics": True,
        "enable_hashing": True,
        "enable_versions": True,
        "enable_history": True
    },
    "paths": {
        "output_dir": "ml/verification",
        "logs_dir": "ml/verification/logs",
        "reports_dir": "ml/verification/reports",
        "statistics_dir": "ml/verification/statistics",
        "metadata_dir": "ml/verification/metadata",
        "hashes_dir": "ml/verification/hashes",
        "versions_dir": "ml/verification/versions",
        "cache_dir": "ml/verification/cache",
        "history_dir": "ml/verification/history",
        "charts_dir": "ml/verification/charts",
        "verification_history_csv": "ml/verification/history/verification_history.csv",
        "verification_statistics_json": "ml/verification/statistics/verification_statistics.json",
        "verification_report_md": "ml/verification/reports/verification_report.md",
        "verification_metadata_json": "ml/verification/metadata/verification_metadata.json",
        "verification_hashes_json": "ml/verification/hashes/verification_hashes.json",
        "verification_versions_json": "ml/verification/versions/verification_versions.json",
        "verification_pipeline_log": "ml/verification/logs/verification_pipeline.log"
    }
}

class VerificationConfig:
    """
    Configuration manager for the Phase 9 Live News Verification Engine.
    """
    def __init__(self, config_path: str = "ml/verification/verification_config.yaml") -> None:
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
    def timeout(self) -> int:
        return int(self._config.get("verification", {}).get("timeout", 10))

    @property
    def retry_count(self) -> int:
        return int(self._config.get("verification", {}).get("retry_count", 3))

    @property
    def rate_limit_delay(self) -> float:
        return float(self._config.get("verification", {}).get("rate_limit_delay", 1.0))

    @property
    def max_results_per_provider(self) -> int:
        return int(self._config.get("verification", {}).get("max_results_per_provider", 5))

    @property
    def similarity_threshold(self) -> float:
        return float(self._config.get("verification", {}).get("similarity_threshold", 0.4))

    @property
    def evidence_threshold(self) -> float:
        return float(self._config.get("verification", {}).get("evidence_threshold", 0.5))

    @property
    def duplicate_threshold(self) -> float:
        return float(self._config.get("verification", {}).get("duplicate_threshold", 0.85))

    @property
    def cache_expiration(self) -> int:
        return int(self._config.get("verification", {}).get("cache_expiration", 3600))

    @property
    def enable_newsapi(self) -> bool:
        return bool(self._config.get("providers", {}).get("enable_newsapi", True))

    @property
    def enable_gnews(self) -> bool:
        return bool(self._config.get("providers", {}).get("enable_gnews", True))

    @property
    def enable_newsdata(self) -> bool:
        return bool(self._config.get("providers", {}).get("enable_newsdata", True))

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
    def enable_hashing(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_hashing", True))

    @property
    def enable_versions(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_versions", True))

    @property
    def enable_history(self) -> bool:
        return bool(self._config.get("exports", {}).get("enable_history", True))

    def get_path(self, path_key: str) -> str:
        return str(self._config.get("paths", {}).get(path_key, ""))
