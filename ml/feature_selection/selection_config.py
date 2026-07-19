import os
import yaml
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("feature_selection_pipeline")

DEFAULT_CONFIG: Dict[str, Any] = {
    "selectors": {
        "variance": {
            "enabled": True,
            "threshold": 0.001,
        },
        "correlation": {
            "enabled": True,
            "threshold": 0.85,
        },
        "mutual_information": {
            "enabled": True,
            "top_k_dense": 15,
            "top_k_sparse": 150,
            "sub_sample_size": 10000,
        },
        "chi_square": {
            "enabled": True,
            "top_k_dense": 15,
            "top_k_sparse": 200,
        },
        "random_forest": {
            "enabled": True,
            "n_estimators": 50,
            "max_depth": 10,
            "random_state": 42,
            "top_k": 100,
            "pre_selected_pool_size": 300,
        },
        "rfe": {
            "enabled": True,
            "n_features_to_select": 30,
            "step": 5,
            "random_state": 42,
            "pre_selected_pool_size": 80,
        }
    },
    "merger": {
        "strategy": "voting",
        "voting_threshold": 2,
    },
    "pipeline": {
        "random_state": 42,
        "batch_size": 5000,
    },
    "paths": {
        "input_csv": "ml/feature_engineering/processed/feature_dataset_v1.csv",
        "input_tfidf_matrix": "ml/feature_engineering/processed/tfidf_matrix.joblib",
        "input_tfidf_vectorizer": "ml/feature_engineering/processed/tfidf_vectorizer.joblib",
        "processed_dir": "ml/feature_selection/processed",
        "models_dir": "ml/feature_selection/models",
        "reports_dir": "ml/feature_selection/reports",
        "statistics_dir": "ml/feature_selection/statistics",
        "versions_dir": "ml/feature_selection/versions",
        "logs_dir": "ml/feature_selection/logs",
        "output_csv": "ml/feature_selection/processed/selected_feature_dataset_v1.csv",
        "selected_names": "ml/feature_selection/processed/selected_feature_names.json",
        "feature_rankings": "ml/feature_selection/processed/feature_rankings.json",
        "summary_file": "ml/feature_selection/processed/feature_selection_summary.json",
        "statistics_file": "ml/feature_selection/statistics/selection_statistics.json",
        "versions_file": "ml/feature_selection/statistics/selection_versions.json",
        "report_file": "ml/feature_selection/reports/feature_selection_report.md",
        "hash_file": "ml/feature_selection/statistics/selection_hash.json",
    }
}

class SelectionConfig:
    """
    Configuration manager for the Phase 5 Feature Selection module.
    Loads and validates the configuration from YAML format.
    """
    def __init__(self, config_path: str = "config/selection_config.yaml") -> None:
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

    def get_selector_setting(self, selector_name: str, key: str, default: Any = None) -> Any:
        """Helper to get selector-specific configuration settings."""
        return self._config.get("selectors", {}).get(selector_name, {}).get(key, default)

    def is_selector_enabled(self, selector_name: str) -> bool:
        """Returns True if the specified feature selector is enabled."""
        return bool(self.get_selector_setting(selector_name, "enabled", False))

    @property
    def merger_strategy(self) -> str:
        """Gets selection merger strategy ('voting', 'union', or 'intersection')."""
        return str(self._config.get("merger", {}).get("strategy", "voting"))

    @property
    def merger_voting_threshold(self) -> int:
        """Gets the minimum votes required for a feature to be kept (voting strategy)."""
        return int(self._config.get("merger", {}).get("voting_threshold", 2))

    @property
    def random_state(self) -> int:
        """Gets the global random state for reproducible calculations."""
        return int(self._config.get("pipeline", {}).get("random_state", 42))

    @property
    def batch_size(self) -> int:
        """Gets the pipeline batch size."""
        return int(self._config.get("pipeline", {}).get("batch_size", 5000))

    def get_path(self, path_key: str) -> str:
        """Resolves file or directory paths."""
        path_val = self._config.get("paths", {}).get(path_key, "")
        if not path_val:
            # Try to resolve from default config paths
            path_val = DEFAULT_CONFIG.get("paths", {}).get(path_key, "")
        return str(path_val)
