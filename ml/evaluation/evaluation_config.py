import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger("model_evaluation_pipeline")

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
        "training_registry_json": "ml/training/registry.json",
        "training_versions_json": "ml/training/versions/training_versions.json"
    },
    "outputs": {
        "reports_dir": "ml/evaluation/reports",
        "statistics_dir": "ml/evaluation/statistics",
        "metadata_dir": "ml/evaluation/metadata",
        "hashes_dir": "ml/evaluation/hashes",
        "versions_dir": "ml/evaluation/versions",
        "leaderboard_dir": "ml/evaluation/leaderboard",
        "comparison_dir": "ml/evaluation/comparison",
        "logs_dir": "ml/evaluation/logs",
        "predictions_dir": "ml/evaluation/predictions",
        "classification_reports_dir": "ml/evaluation/classification_reports",
        "confusion_matrices_dir": "ml/evaluation/confusion_matrices",
        "roc_curves_dir": "ml/evaluation/roc_curves",
        "precision_recall_curves_dir": "ml/evaluation/precision_recall_curves",
        "charts_dir": "ml/evaluation/charts",
        
        "evaluation_report_file": "ml/evaluation/reports/evaluation_report.md",
        "evaluation_statistics_file": "ml/evaluation/statistics/evaluation_statistics.json",
        "leaderboard_csv_file": "ml/evaluation/leaderboard/leaderboard.csv",
        "leaderboard_json_file": "ml/evaluation/leaderboard/leaderboard.json",
        "leaderboard_md_file": "ml/evaluation/leaderboard/leaderboard.md",
        "comparison_csv_file": "ml/evaluation/comparison/model_comparison.csv",
        "comparison_json_file": "ml/evaluation/comparison/model_comparison.json",
        "comparison_md_file": "ml/evaluation/comparison/model_comparison.md",
        "best_model_file": "ml/evaluation/best_model.json",
        "versions_file": "ml/evaluation/versions/evaluation_versions.json",
        "hashes_file": "ml/evaluation/hashes/evaluation_hashes.json"
    },
    "visualizations": {
        "enable_charts": True,
        "enable_roc": True,
        "enable_pr_curve": True
    },
    "best_model_selection": {
        "selection_metric": "weighted_score",
        "weights": {
            "f1_score": 0.40,
            "roc_auc": 0.30,
            "precision": 0.15,
            "recall": 0.10,
            "prediction_speed": 0.05
        }
    }
}

class EvaluationConfig:
    """
    Configuration manager for Phase 7 Model Evaluation.
    Loads, merges, and validates configuration from evaluation_config.yaml.
    """
    def __init__(self, config_path: str = "ml/evaluation/evaluation_config.yaml") -> None:
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

    @property
    def enable_charts(self) -> bool:
        """Checks if charts are enabled."""
        return bool(self._config.get("visualizations", {}).get("enable_charts", True))

    @property
    def enable_roc(self) -> bool:
        """Checks if ROC visualization is enabled."""
        return bool(self._config.get("visualizations", {}).get("enable_roc", True))

    @property
    def enable_pr_curve(self) -> bool:
        """Checks if Precision-Recall curve is enabled."""
        return bool(self._config.get("visualizations", {}).get("enable_pr_curve", True))

    @property
    def selection_metric(self) -> str:
        """Gets the metric key used to select the best model."""
        return str(self._config.get("best_model_selection", {}).get("selection_metric", "weighted_score"))

    @property
    def selection_weights(self) -> Dict[str, float]:
        """Gets the dictionary of metric weights for selection."""
        weights = self._config.get("best_model_selection", {}).get("weights", {})
        return {str(k): float(v) for k, v in weights.items()}

    def get_input_path(self, key: str) -> str:
        """Gets input path for a given key."""
        return str(self._config.get("inputs", {}).get(key, ""))

    def get_output_path(self, key: str) -> str:
        """Gets output path/file for a given key."""
        return str(self._config.get("outputs", {}).get(key, ""))

    def get_output_dir(self, key: str) -> str:
        """Gets output directory for a given key."""
        return str(self._config.get("outputs", {}).get(key, ""))
