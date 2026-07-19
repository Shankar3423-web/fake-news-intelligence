"""
retraining_config.py
Configuration manager for Phase 12: Automatic Model Retraining System.
Loads and merges configuration from retraining_config.yaml.
"""
import os
import yaml
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("retraining_pipeline")

# ── Defaults ──────────────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "approved_feedback": {
        "csv_path": "ml/admin_review/approved/approved_feedback.csv",
        "required_columns": ["Feedback ID", "Timestamp", "Prediction", "Decision", "Feedback"],
        "valid_labels": [0, 1],
        "min_records": 1,
    },
    "training_dataset": {
        "csv_path": "ml/feature_selection/processed/selected_feature_dataset_v1.csv",
        "feature_names_json": "ml/feature_selection/processed/selected_feature_names.json",
        "label_column": "label",
        "id_column": "id",
    },
    "raw_dataset": {
        "csv_path": "ml/dataset/processed/fake_news_dataset.csv",
        "text_column": "text",
        "label_column": "label",
    },
    "phase_configs": {
        "preprocessing": "config/preprocessing_config.yaml",
        "feature_engineering": "config/feature_config.yaml",
        "feature_selection": "config/selection_config.yaml",
        "training": "ml/training/training_config.yaml",
        "evaluation": "ml/evaluation/evaluation_config.yaml",
    },
    "candidate": {
        "models_dir": "ml/retraining/candidate_models",
        "production_models_dir": "ml/retraining/production_models",
        "registry_file": "ml/retraining/registry/retraining_registry.json",
    },
    "outputs": {
        "logs_dir": "ml/retraining/logs",
        "reports_dir": "ml/retraining/reports",
        "metadata_dir": "ml/retraining/metadata",
        "statistics_dir": "ml/retraining/statistics",
        "versions_dir": "ml/retraining/versions",
        "hashes_dir": "ml/retraining/hashes",
        "charts_dir": "ml/retraining/charts",
        "registry_dir": "ml/retraining/registry",
    },
    "output_files": {
        "retraining_report": "ml/retraining/reports/retraining_report.md",
        "comparison_report": "ml/retraining/reports/comparison_report.md",
        "retraining_statistics": "ml/retraining/statistics/retraining_statistics.json",
        "retraining_metadata": "ml/retraining/metadata/retraining_metadata.json",
        "retraining_hashes": "ml/retraining/hashes/retraining_hashes.json",
        "retraining_versions": "ml/retraining/versions/retraining_versions.json",
        "deployment_decision": "ml/retraining/reports/deployment_decision.json",
        "retraining_log": "ml/retraining/logs/retraining_pipeline.log",
    },
    "acceptance_policy": {
        "primary_metric": "f1_score",
        "minimum_improvement_delta": 0.0,
        "minimum_thresholds": {
            "f1_score": 0.60,
            "accuracy": 0.60,
        },
        "promote_on_tie": False,
        "weighted_comparison": {
            "enabled": True,
            "weights": {
                "f1_score": 0.35,
                "roc_auc": 0.25,
                "accuracy": 0.15,
                "precision": 0.10,
                "recall": 0.10,
                "mcc": 0.05,
            },
        },
        "inference_time_threshold_sec": 5.0,
        "memory_threshold_mb": 512.0,
    },
    "merge": {
        "duplicate_strategy": "last",
        "shuffle": True,
        "random_state": 42,
    },
    "random_state": 42,
    "feedback_label_mapping": {
        "decisions_to_label": {
            "REAL": 0,
            "FAKE": 1,
            "SUSPECTED REAL": 0,
            "SUSPECTED FAKE": 1,
        }
    },
}


class RetrainingConfig:
    """
    Configuration manager for Phase 12: Automatic Model Retraining System.

    Loads YAML config and merges with defaults so all keys are always
    available regardless of partial user overrides.
    """

    def __init__(
        self, config_path: str = "ml/retraining/retraining_config.yaml"
    ) -> None:
        self.config_path: str = config_path
        self._config: Dict[str, Any] = self._load_config()

    # ── Loading ───────────────────────────────────────────────────────────────────────────────────
    def _load_config(self) -> Dict[str, Any]:
        """Loads YAML with deep-merge fallback to DEFAULT_CONFIG."""
        if not os.path.exists(self.config_path):
            logger.warning(
                "Retraining config not found at %s. Using defaults.", self.config_path
            )
            return _deep_merge(DEFAULT_CONFIG, {})

        try:
            with open(self.config_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            return _deep_merge(DEFAULT_CONFIG, loaded)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Error loading retraining config from %s: %s. Using defaults.",
                self.config_path,
                exc,
            )
            return _deep_merge(DEFAULT_CONFIG, {})

    # ── Convenience accessors ─────────────────────────────────────────────────────────────────────
    def get(self, *keys: str, default: Any = None) -> Any:
        """Navigates nested keys and returns the value (or *default* if missing)."""
        node: Any = self._config
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    def get_output_dir(self, key: str) -> str:
        """Returns an output directory path for *key*."""
        return str(self._config.get("outputs", {}).get(key, ""))

    def get_output_file(self, key: str) -> str:
        """Returns an output file path for *key*."""
        return str(self._config.get("output_files", {}).get(key, ""))

    def get_approved_feedback_path(self) -> str:
        return str(self._config["approved_feedback"]["csv_path"])

    def get_training_dataset_path(self) -> str:
        return str(self._config["training_dataset"]["csv_path"])

    def get_feature_names_path(self) -> str:
        return str(self._config["training_dataset"]["feature_names_json"])

    def get_phase_config_path(self, phase: str) -> str:
        return str(self._config["phase_configs"].get(phase, ""))

    def get_candidate_models_dir(self) -> str:
        return str(self._config["candidate"]["models_dir"])

    def get_production_models_dir(self) -> str:
        return str(self._config["candidate"]["production_models_dir"])

    def get_registry_file(self) -> str:
        return str(self._config["candidate"]["registry_file"])

    # ── Acceptance policy ─────────────────────────────────────────────────────────────────────────
    @property
    def primary_metric(self) -> str:
        return str(self._config["acceptance_policy"]["primary_metric"])

    @property
    def minimum_improvement_delta(self) -> float:
        return float(self._config["acceptance_policy"]["minimum_improvement_delta"])

    @property
    def minimum_thresholds(self) -> Dict[str, float]:
        return dict(self._config["acceptance_policy"]["minimum_thresholds"])

    @property
    def promote_on_tie(self) -> bool:
        return bool(self._config["acceptance_policy"]["promote_on_tie"])

    @property
    def weighted_comparison_enabled(self) -> bool:
        return bool(
            self._config["acceptance_policy"]["weighted_comparison"]["enabled"]
        )

    @property
    def weighted_comparison_weights(self) -> Dict[str, float]:
        return dict(
            self._config["acceptance_policy"]["weighted_comparison"]["weights"]
        )

    # ── Feedback / merge ──────────────────────────────────────────────────────────────────────────
    @property
    def required_feedback_columns(self) -> List[str]:
        return list(self._config["approved_feedback"]["required_columns"])

    @property
    def valid_labels(self) -> List[int]:
        return list(self._config["approved_feedback"]["valid_labels"])

    @property
    def min_approved_records(self) -> int:
        return int(self._config["approved_feedback"]["min_records"])

    @property
    def feedback_label_mapping(self) -> Dict[str, int]:
        return dict(
            self._config["feedback_label_mapping"]["decisions_to_label"]
        )

    @property
    def random_state(self) -> int:
        return int(self._config.get("random_state", 42))

    @property
    def merge_shuffle(self) -> bool:
        return bool(self._config["merge"]["shuffle"])

    @property
    def merge_duplicate_strategy(self) -> str:
        return str(self._config["merge"]["duplicate_strategy"])

    # ── All config ────────────────────────────────────────────────────────────────────────────────
    def as_dict(self) -> Dict[str, Any]:
        """Returns the merged configuration as a plain dictionary."""
        return dict(self._config)


# ── Helpers ───────────────────────────────────────────────────────────────────────────────────────
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges *override* into *base* (non-destructive copy)."""
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
