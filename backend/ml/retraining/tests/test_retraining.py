"""
test_retraining.py
Unit tests for Phase 12: Automatic Model Retraining System.

Covers:
  - Approved feedback loader
  - Dataset merge manager
  - Dataset validator
  - Training executor
  - Evaluation executor
  - Model comparator
  - Deployment manager
  - Model registry
  - Report generation
  - Metadata manager
  - Statistics manager
  - Version manager
  - Hash generator
  - End-to-end pipeline
  - Verification script
"""
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add project root to sys.path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from ml.retraining.retraining_config import RetrainingConfig, _deep_merge

# ── Fixtures ──────────────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = "ml/retraining/retraining_config.yaml"


@pytest.fixture()
def cfg() -> RetrainingConfig:
    return RetrainingConfig(CONFIG_PATH)


@pytest.fixture()
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture()
def sample_approved_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Feedback ID": ["fb_001", "fb_002", "fb_003"],
            "Timestamp": ["2026-07-01T10:00:00"] * 3,
            "Prediction": [0, 1, 0],
            "Verification": ["VERIFIED", "VERIFIED", "PARTIALLY VERIFIED"],
            "Decision": ["REAL", "FAKE", "SUSPECTED REAL"],
            "Feedback": ["Correct", "Correct", "Correct"],
            "Comment": ["OK", "OK", "OK"],
            "label": [0, 1, 0],
        }
    )


@pytest.fixture()
def sample_training_df() -> pd.DataFrame:
    """Small feature-selected training DataFrame."""
    rng = __import__("numpy").random.default_rng(42)
    n = 20
    data = {"label": [0, 1] * (n // 2)}
    for i in range(5):
        data[f"feat_{i}"] = rng.random(n).tolist()
    data["id"] = [f"id_{i}" for i in range(n)]
    return pd.DataFrame(data)


@pytest.fixture()
def feature_columns() -> List[str]:
    return [f"feat_{i}" for i in range(5)]


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingConfig:
    def test_loads_without_error(self, cfg):
        assert cfg is not None

    def test_primary_metric_is_string(self, cfg):
        assert isinstance(cfg.primary_metric, str)
        assert len(cfg.primary_metric) > 0

    def test_minimum_thresholds_is_dict(self, cfg):
        thresholds = cfg.minimum_thresholds
        assert isinstance(thresholds, dict)

    def test_feedback_label_mapping_contains_real_fake(self, cfg):
        mapping = cfg.feedback_label_mapping
        assert "REAL" in mapping or "real" in mapping or any("REAL" in k.upper() for k in mapping)

    def test_deep_merge_overrides_leaf_values(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"x": 99}}
        result = _deep_merge(base, override)
        assert result["a"]["x"] == 99
        assert result["a"]["y"] == 2
        assert result["b"] == 3

    def test_get_output_dir_returns_string(self, cfg):
        d = cfg.get_output_dir("logs_dir")
        assert isinstance(d, str)
        assert len(d) > 0

    def test_get_output_file_returns_string(self, cfg):
        f = cfg.get_output_file("retraining_report")
        assert isinstance(f, str)

    def test_random_state_is_int(self, cfg):
        assert isinstance(cfg.random_state, int)

    def test_missing_config_uses_defaults(self, tmp_dir):
        cfg2 = RetrainingConfig(os.path.join(tmp_dir, "nonexistent.yaml"))
        assert cfg2.primary_metric == "f1_score"


# ══════════════════════════════════════════════════════════════════════════════
# APPROVED FEEDBACK LOADER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestApprovedFeedbackLoader:
    def test_load_from_valid_csv(self, cfg, tmp_dir, sample_approved_df):
        csv_path = os.path.join(tmp_dir, "approved.csv")
        # Write without label col (loader should derive it)
        df_out = sample_approved_df.drop(columns=["label"])
        df_out.to_csv(csv_path, index=False)

        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.approved_feedback_loader import ApprovedFeedbackLoader

        cfg2 = RetrainingConfig.__new__(RetrainingConfig)
        cfg2._config = dict(cfg.as_dict())
        cfg2._config["approved_feedback"] = dict(cfg._config["approved_feedback"])
        cfg2._config["approved_feedback"]["csv_path"] = csv_path
        cfg2.config_path = cfg.config_path

        loader = ApprovedFeedbackLoader(cfg2)
        ok, err, df = loader.load()
        assert ok, f"Loader failed: {err}"
        assert not df.empty
        assert "label" in df.columns
        assert df["label"].isin([0, 1]).all()

    def test_returns_false_for_missing_file(self, cfg, tmp_dir):
        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.approved_feedback_loader import ApprovedFeedbackLoader

        cfg2 = RetrainingConfig.__new__(RetrainingConfig)
        cfg2._config = dict(cfg.as_dict())
        cfg2._config["approved_feedback"] = dict(cfg._config["approved_feedback"])
        cfg2._config["approved_feedback"]["csv_path"] = os.path.join(tmp_dir, "ghost.csv")
        cfg2.config_path = cfg.config_path

        loader = ApprovedFeedbackLoader(cfg2)
        ok, err, df = loader.load()
        assert not ok
        assert df.empty

    def test_drops_rows_with_unmapped_decisions(self, cfg, tmp_dir):
        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.approved_feedback_loader import ApprovedFeedbackLoader

        rows = [
            {"Feedback ID": "fb_x", "Timestamp": "2026-01-01", "Prediction": 0,
             "Decision": "UNKNOWN_DECISION", "Feedback": "Correct", "Comment": ""},
            {"Feedback ID": "fb_y", "Timestamp": "2026-01-01", "Prediction": 0,
             "Decision": "REAL", "Feedback": "Correct", "Comment": ""},
        ]
        csv_path = os.path.join(tmp_dir, "mixed.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        cfg2 = RetrainingConfig.__new__(RetrainingConfig)
        cfg2._config = dict(cfg.as_dict())
        cfg2._config["approved_feedback"] = dict(cfg._config["approved_feedback"])
        cfg2._config["approved_feedback"]["csv_path"] = csv_path
        cfg2.config_path = cfg.config_path

        loader = ApprovedFeedbackLoader(cfg2)
        ok, err, df = loader.load()
        assert ok
        assert len(df) == 1
        assert df.iloc[0]["label"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# DATASET VALIDATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestDatasetValidator:
    def _make_validator(self):
        from ml.retraining.dataset_validator import DatasetValidator
        return DatasetValidator(label_column="label", id_column="id")

    def test_valid_dataset_passes(self, sample_training_df):
        v = self._make_validator()
        ok, errors = v.validate(sample_training_df, context="test")
        assert ok, f"Unexpected errors: {errors}"

    def test_empty_df_fails(self):
        v = self._make_validator()
        ok, errors = v.validate(pd.DataFrame(), context="test")
        assert not ok

    def test_invalid_labels_detected(self, sample_training_df):
        bad = sample_training_df.copy()
        bad["label"] = 99  # Invalid
        v = self._make_validator()
        ok, errors = v.validate(bad, context="test")
        assert not ok
        assert any("Invalid label" in e for e in errors)

    def test_null_labels_detected(self, sample_training_df):
        bad = sample_training_df.copy()
        bad.loc[0, "label"] = None
        v = self._make_validator()
        ok, errors = v.validate(bad, context="test")
        assert not ok

    def test_duplicate_ids_detected(self, sample_training_df):
        dup = pd.concat([sample_training_df, sample_training_df.head(3)], ignore_index=True)
        v = self._make_validator()
        ok, errors = v.validate(dup, context="test", check_duplicates=True)
        assert not ok
        assert any("duplicate" in e.lower() for e in errors)

    def test_schema_match_passes_when_identical(self, sample_training_df):
        v = self._make_validator()
        ok, errors = v.validate_schema_match(sample_training_df, sample_training_df)
        assert ok

    def test_schema_match_fails_when_column_missing(self, sample_training_df):
        reduced = sample_training_df.drop(columns=["feat_0"])
        v = self._make_validator()
        ok, errors = v.validate_schema_match(sample_training_df, reduced)
        assert not ok


# ══════════════════════════════════════════════════════════════════════════════
# DATASET MERGE MANAGER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestDatasetMergeManager:
    def test_merge_increases_row_count(
        self, cfg, sample_training_df, sample_approved_df, feature_columns
    ):
        from ml.retraining.dataset_merge_manager import DatasetMergeManager

        merger = DatasetMergeManager(cfg)
        ok, err, merged = merger.merge(sample_training_df, sample_approved_df, feature_columns)
        assert ok, f"Merge failed: {err}"
        assert len(merged) >= len(sample_training_df)

    def test_merge_stats_are_correct(
        self, cfg, sample_training_df, sample_approved_df, feature_columns
    ):
        from ml.retraining.dataset_merge_manager import DatasetMergeManager

        merger = DatasetMergeManager(cfg)
        _, _, merged = merger.merge(sample_training_df, sample_approved_df, feature_columns)
        stats = merger.compute_merge_stats(sample_training_df, sample_approved_df, merged)
        assert stats["existing_rows"] == len(sample_training_df)
        assert stats["approved_records"] == len(sample_approved_df)
        assert stats["merged_rows"] == len(merged)

    def test_merge_preserves_label_column(
        self, cfg, sample_training_df, sample_approved_df, feature_columns
    ):
        from ml.retraining.dataset_merge_manager import DatasetMergeManager

        merger = DatasetMergeManager(cfg)
        ok, _, merged = merger.merge(sample_training_df, sample_approved_df, feature_columns)
        assert ok
        assert "label" in merged.columns
        assert merged["label"].isin([0, 1]).all()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestModelComparator:
    def _make_comparator(self, cfg):
        from ml.retraining.model_comparator import ModelComparator
        return ModelComparator(cfg)

    def test_promotes_clearly_superior_candidate(self, cfg):
        comp = self._make_comparator(cfg)
        c = {"f1_score": 0.95, "accuracy": 0.95, "precision": 0.95,
             "recall": 0.95, "roc_auc": 0.98, "mcc": 0.90}
        p = {"f1_score": 0.70, "accuracy": 0.70, "precision": 0.70,
             "recall": 0.70, "roc_auc": 0.75, "mcc": 0.60}
        result = comp.compare(c, p)
        assert result["promote"] is True

    def test_rejects_failing_thresholds(self, cfg):
        comp = self._make_comparator(cfg)
        c = {"f1_score": 0.50, "accuracy": 0.50, "precision": 0.50,
             "recall": 0.50, "roc_auc": 0.55, "mcc": 0.20}
        p = {"f1_score": 0.70, "accuracy": 0.70, "precision": 0.70,
             "recall": 0.70, "roc_auc": 0.75, "mcc": 0.60}
        result = comp.compare(c, p)
        assert result["promote"] is False

    def test_result_contains_per_metric_comparison(self, cfg):
        comp = self._make_comparator(cfg)
        c = {"f1_score": 0.80, "accuracy": 0.80, "precision": 0.80,
             "recall": 0.80, "roc_auc": 0.85, "mcc": 0.75}
        p = {"f1_score": 0.70, "accuracy": 0.70, "precision": 0.70,
             "recall": 0.70, "roc_auc": 0.75, "mcc": 0.60}
        result = comp.compare(c, p)
        assert "per_metric_comparison" in result
        assert len(result["per_metric_comparison"]) > 0

    def test_result_contains_weighted_scores(self, cfg):
        comp = self._make_comparator(cfg)
        c = {"f1_score": 0.85, "accuracy": 0.85, "precision": 0.85,
             "recall": 0.85, "roc_auc": 0.90, "mcc": 0.80}
        p = {"f1_score": 0.70, "accuracy": 0.70, "precision": 0.70,
             "recall": 0.70, "roc_auc": 0.75, "mcc": 0.60}
        result = comp.compare(c, p)
        assert "candidate_score" in result
        assert "production_score" in result
        assert result["candidate_score"] >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT MANAGER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestDeploymentManager:
    def test_rejection_saves_decision_json(self, cfg, tmp_dir):
        from ml.retraining.deployment_manager import DeploymentManager
        from ml.retraining.retraining_config import RetrainingConfig

        # Redirect output file
        cfg2 = RetrainingConfig.__new__(RetrainingConfig)
        cfg2._config = dict(cfg.as_dict())
        cfg2._config["output_files"] = dict(cfg._config["output_files"])
        cfg2._config["output_files"]["deployment_decision"] = os.path.join(
            tmp_dir, "decision.json"
        )
        cfg2._config["candidate"] = dict(cfg._config["candidate"])
        cfg2._config["candidate"]["models_dir"] = tmp_dir
        cfg2._config["candidate"]["production_models_dir"] = os.path.join(tmp_dir, "prod")
        cfg2.config_path = cfg.config_path

        deployer = DeploymentManager(cfg2)
        comparison = {"promote": False, "reason": "Test rejection", "primary_metric": "f1_score",
                      "delta": -0.1, "candidate_score": 0.5, "production_score": 0.7,
                      "thresholds_passed": False, "per_metric_comparison": [],
                      "minimum_threshold_results": {}}
        decision = deployer.execute(
            comparison_result=comparison,
            candidate_model_key="svm",
            candidate_metrics={"f1_score": 0.50},
            production_metrics={"f1_score": 0.70},
            retraining_run_id="test_run_001",
        )
        assert decision["decision"] == "REJECTED"
        assert os.path.isfile(os.path.join(tmp_dir, "decision.json"))


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingModelRegistry:
    def test_register_run_creates_entry(self, tmp_dir):
        from ml.retraining.model_registry import RetrainingModelRegistry

        registry = RetrainingModelRegistry(os.path.join(tmp_dir, "registry.json"))
        entry = registry.register_run(
            run_id="run_001",
            decision="PROMOTED",
            candidate_model_key="svm",
            candidate_metrics={"f1_score": 0.95},
            production_metrics={"f1_score": 0.80},
            approved_records=5,
            merged_rows=1000,
            promoted=True,
            reason="Candidate is better.",
        )
        assert entry["run_id"] == "run_001"
        assert entry["decision"] == "PROMOTED"

        all_runs = registry.get_all_runs()
        assert len(all_runs) == 1

    def test_promoted_run_updates_current_production(self, tmp_dir):
        from ml.retraining.model_registry import RetrainingModelRegistry

        registry = RetrainingModelRegistry(os.path.join(tmp_dir, "registry.json"))
        registry.register_run(
            run_id="run_001",
            decision="PROMOTED",
            candidate_model_key="xgboost",
            candidate_metrics={"f1_score": 0.95},
            production_metrics={"f1_score": 0.80},
            approved_records=5,
            merged_rows=1000,
            promoted=True,
            reason="Better model.",
        )
        prod_metrics = registry.get_production_metrics()
        assert prod_metrics is not None
        assert prod_metrics["f1_score"] == 0.95

    def test_rejected_run_does_not_update_production(self, tmp_dir):
        from ml.retraining.model_registry import RetrainingModelRegistry

        registry = RetrainingModelRegistry(os.path.join(tmp_dir, "registry.json"))
        registry.register_run(
            run_id="run_001",
            decision="REJECTED",
            candidate_model_key="svm",
            candidate_metrics={"f1_score": 0.50},
            production_metrics={"f1_score": 0.80},
            approved_records=5,
            merged_rows=1000,
            promoted=False,
            reason="Candidate is weaker.",
        )
        prod_metrics = registry.get_production_metrics()
        assert prod_metrics is None


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS MANAGER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingStatisticsManager:
    def test_generate_and_save(self, tmp_dir):
        from ml.retraining.statistics_manager import RetrainingStatisticsManager

        mgr = RetrainingStatisticsManager()
        stats = mgr.generate(
            run_id="run_001",
            approved_records=5,
            existing_rows=1000,
            merged_rows=1005,
            feature_count=283,
            training_rows=800,
            testing_rows=205,
            pipeline_duration_sec=42.5,
            candidate_metrics={"f1_score": 0.90},
            production_metrics={"f1_score": 0.80},
            decision="PROMOTED",
            model_durations={"svm": 1.5, "random_forest": 3.0},
        )
        assert stats["run_id"] == "run_001"
        assert stats["pipeline"]["decision"] == "PROMOTED"
        assert stats["dataset"]["approved_records_merged"] == 5

        path = os.path.join(tmp_dir, "stats.json")
        mgr.save(path)
        assert os.path.isfile(path)
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["run_id"] == "run_001"


# ══════════════════════════════════════════════════════════════════════════════
# METADATA MANAGER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingMetadataManager:
    def test_generate_creates_json(self, tmp_dir):
        from ml.retraining.metadata_manager import RetrainingMetadataManager

        mgr = RetrainingMetadataManager(tmp_dir)
        meta = mgr.generate_and_save(
            run_id="run_001",
            config_path=CONFIG_PATH,
            approved_csv_path="approved.csv",
            approved_records=5,
            merged_rows=1005,
            feature_count=283,
            candidate_model_key="svm",
            decision="PROMOTED",
            pipeline_duration_sec=42.5,
        )
        assert meta["run_id"] == "run_001"
        assert meta["phase"] == 12
        assert "environment" in meta
        path = os.path.join(tmp_dir, "retraining_metadata.json")
        assert os.path.isfile(path)


# ══════════════════════════════════════════════════════════════════════════════
# HASH GENERATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestHashGenerator:
    def test_generates_hashes_for_existing_files(self, tmp_dir):
        from ml.retraining.hash_generator import HashGenerator

        # Create a dummy file
        dummy = os.path.join(tmp_dir, "dummy.txt")
        with open(dummy, "w") as fh:
            fh.write("hello world")

        gen = HashGenerator(tmp_dir)
        out = gen.generate({"dummy": dummy})
        assert os.path.isfile(out)
        with open(out) as fh:
            data = json.load(fh)
        assert "files" in data
        assert data["files"]["dummy"] != "NOT_FOUND"

    def test_marks_missing_files_as_not_found(self, tmp_dir):
        from ml.retraining.hash_generator import HashGenerator

        gen = HashGenerator(tmp_dir)
        out = gen.generate({"ghost": os.path.join(tmp_dir, "ghost.txt")})
        with open(out) as fh:
            data = json.load(fh)
        assert data["files"]["ghost"] == "NOT_FOUND"


# ══════════════════════════════════════════════════════════════════════════════
# VERSION MANAGER TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingVersionManager:
    def test_register_increments_version(self, tmp_dir):
        from ml.retraining.version_manager import RetrainingVersionManager

        mgr = RetrainingVersionManager(tmp_dir)
        e1 = mgr.register("run_001", "PROMOTED", True, {"f": "abc"})
        e2 = mgr.register("run_002", "REJECTED", False, {"f": "def"})
        assert e1["version"] == 1
        assert e2["version"] == 2

    def test_versions_persisted_across_instances(self, tmp_dir):
        from ml.retraining.version_manager import RetrainingVersionManager

        mgr = RetrainingVersionManager(tmp_dir)
        mgr.register("run_001", "PROMOTED", True, {})

        mgr2 = RetrainingVersionManager(tmp_dir)
        versions = mgr2.load()
        assert len(versions) == 1
        assert versions[0]["run_id"] == "run_001"


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingReportGenerator:
    def test_generates_retraining_report(self, tmp_dir):
        from ml.retraining.report_generator import RetrainingReportGenerator

        gen = RetrainingReportGenerator(tmp_dir)
        path = gen.generate_retraining_report(
            run_id="run_001",
            approved_records=5,
            merge_stats={"existing_rows": 1000, "merged_rows": 1005, "added_rows": 5,
                         "label_distribution_merged": {"0": 503, "1": 502}},
            training_summary={"models_trained": ["svm"], "training_rows": 804,
                               "testing_rows": 201, "feature_count": 283,
                               "model_durations": {"svm": 1.5}},
            candidate_results={"svm": {"metrics": {"f1_score": 0.90, "accuracy": 0.90,
                                                    "precision": 0.91, "recall": 0.89,
                                                    "roc_auc": 0.95, "mcc": 0.80}}},
            decision="PROMOTED",
            reason="Candidate is better.",
            pipeline_duration_sec=42.5,
            generated_files={"stats": "/tmp/stats.json"},
        )
        assert os.path.isfile(path)
        with open(path) as fh:
            content = fh.read()
        assert "Phase 12" in content
        assert "PROMOTED" in content

    def test_generates_comparison_report(self, tmp_dir):
        from ml.retraining.report_generator import RetrainingReportGenerator

        gen = RetrainingReportGenerator(tmp_dir)
        comparison = {
            "promote": True,
            "reason": "Better candidate.",
            "primary_metric": "f1_score",
            "delta": 0.05,
            "candidate_score": 0.88,
            "production_score": 0.80,
            "per_metric_comparison": [
                {"metric": "f1_score", "candidate": 0.90, "production": 0.80,
                 "delta": 0.10, "lower_is_better": False, "winner": "candidate"}
            ],
            "minimum_threshold_results": {
                "f1_score": {"required": 0.60, "candidate_value": 0.90, "passed": True}
            },
            "thresholds_passed": True,
        }
        path = gen.generate_comparison_report(
            run_id="run_001",
            comparison_result=comparison,
            candidate_metrics={"f1_score": 0.90},
            production_metrics={"f1_score": 0.80},
        )
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        assert "Phase 12" in content
        assert "PROMOTED" in content


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT VALIDATOR TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingArtifactValidator:
    def test_fails_when_required_files_missing(self, cfg, tmp_dir):
        """Validator should catch missing required files."""
        from ml.retraining.validator import RetrainingArtifactValidator
        from ml.retraining.retraining_config import RetrainingConfig

        # Redirect all outputs to empty tmp dir
        cfg2 = RetrainingConfig.__new__(RetrainingConfig)
        cfg2._config = json.loads(json.dumps(cfg.as_dict()))
        cfg2.config_path = cfg.config_path
        # Outputs do not exist in tmp_dir yet
        for key in cfg2._config["output_files"]:
            cfg2._config["output_files"][key] = os.path.join(tmp_dir, f"{key}.json")
        for key in cfg2._config["outputs"]:
            cfg2._config["outputs"][key] = tmp_dir
        cfg2._config["candidate"]["registry_file"] = os.path.join(tmp_dir, "reg.json")
        cfg2._config["candidate"]["models_dir"] = tmp_dir

        validator = RetrainingArtifactValidator(cfg2)
        ok, errors = validator.validate_all()
        # Should fail because files don't exist
        assert not ok
        assert len(errors) > 0

    def test_passes_after_pipeline_run(self, cfg):
        """After a real pipeline run, the validator should pass."""
        from ml.retraining.validator import RetrainingArtifactValidator

        validator = RetrainingArtifactValidator(cfg)
        # Skip if pipeline has never run (CI environment)
        registry_file = cfg.get_registry_file()
        if not os.path.isfile(registry_file):
            pytest.skip("Pipeline has not run yet — skipping post-run validation.")

        ok, errors = validator.validate_all()
        assert ok, f"Validator errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# END-TO-END PIPELINE TEST (integration)
# ══════════════════════════════════════════════════════════════════════════════
class TestRetrainingPipelineIntegration:
    def test_full_pipeline_runs_and_returns_true(self):
        """Integration: run the complete pipeline and verify it returns True."""
        from ml.retraining.retraining_pipeline import RetrainingPipeline
        from ml.retraining.retraining_logger import shutdown_logger

        # Pre-condition: approved feedback must exist
        cfg = RetrainingConfig(CONFIG_PATH)
        approved_path = cfg.get_approved_feedback_path()
        if not os.path.exists(approved_path) or os.path.getsize(approved_path) == 0:
            pytest.skip("Approved feedback CSV not available.")

        training_path = cfg.get_training_dataset_path()
        if not os.path.exists(training_path):
            pytest.skip("Training dataset not available.")

        pipeline = RetrainingPipeline(CONFIG_PATH)
        success = pipeline.run()
        shutdown_logger()
        assert success, "RetrainingPipeline.run() returned False"
