import os
import yaml
import logging
from typing import Dict, Any, List

logger = logging.getLogger("model_training_pipeline")

DEFAULT_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "split": {
        "test_size": 0.2,
        "stratify": True,
        "shuffle": True
    },
    "inputs": {
        "dataset_csv": "ml/feature_selection/processed/selected_feature_dataset_v1.csv",
        "feature_names_json": "ml/feature_selection/processed/selected_feature_names.json",
        "selection_versions_json": "ml/feature_selection/statistics/selection_versions.json"
    },
    "outputs": {
        "models_dir": "ml/training/models",
        "metadata_dir": "ml/training/metadata",
        "reports_dir": "ml/training/reports",
        "statistics_dir": "ml/training/statistics",
        "versions_dir": "ml/training/versions",
        "hashes_dir": "ml/training/hashes",
        "logs_dir": "ml/training/logs",
        "registry_file": "ml/training/registry.json",
        "statistics_file": "ml/training/training_statistics.json",
        "report_file": "ml/training/training_report.md",
        "versions_file": "ml/training/training_versions.json"
    },
    "models": {
        "logistic_regression": {
            "enabled": True,
            "hyperparameters": {
                "solver": "lbfgs",
                "max_iter": 1000,
                "C": 1.0,
                "random_state": 42
            }
        },
        "svm": {
            "enabled": True,
            "hyperparameters": {
                "C": 1.0,
                "max_iter": 2000,
                "random_state": 42
            }
        },
        "random_forest": {
            "enabled": True,
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 15,
                "random_state": 42,
                "n_jobs": -1
            }
        },
        "xgboost": {
            "enabled": True,
            "hyperparameters": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "random_state": 42,
                "eval_metric": "logloss"
            }
        }
    }
}

class TrainingConfig:
    """
    Configuration manager for the Phase 6 Model Training module.
    Loads, merges, and validates the configuration from training_config.yaml.
    """
    def __init__(self, config_path: str = "ml/training/training_config.yaml") -> None:
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

    @property
    def random_state(self) -> int:
        """Gets the global random state for reproducibility."""
        return int(self._config.get("random_state", 42))

    @property
    def test_size(self) -> float:
        """Gets the split test size."""
        return float(self._config.get("split", {}).get("test_size", 0.2))

    @property
    def stratify(self) -> bool:
        """Gets whether to stratify during split."""
        return bool(self._config.get("split", {}).get("stratify", True))

    @property
    def shuffle(self) -> bool:
        """Gets whether to shuffle during split."""
        return bool(self._config.get("split", {}).get("shuffle", True))

    def is_model_enabled(self, model_key: str) -> bool:
        """Checks if a model is enabled."""
        return bool(self._config.get("models", {}).get(model_key, {}).get("enabled", False))

    def get_model_hyperparameters(self, model_key: str) -> Dict[str, Any]:
        """Gets the hyperparameters for a specific model."""
        return self._config.get("models", {}).get(model_key, {}).get("hyperparameters", {}).copy()

    def get_input_path(self, key: str) -> str:
        """Gets input path for a given key."""
        return str(self._config.get("inputs", {}).get(key, ""))

    def get_output_path(self, key: str) -> str:
        """Gets output path/file for a given key."""
        return str(self._config.get("outputs", {}).get(key, ""))

    def get_output_dir(self, key: str) -> str:
        """Gets output directory for a given key."""
        return str(self._config.get("outputs", {}).get(key, ""))
