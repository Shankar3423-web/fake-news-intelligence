"""
verify_retraining.py
Phase 12 – Automatic Model Retraining System Verification Script.

Verifies:
  1.  Folder structure
  2.  Configuration loading
  3.  Approved dataset loading
  4.  Dataset merge
  5.  Candidate training
  6.  Candidate evaluation
  7.  Model comparison
  8.  Deployment decision
  9.  Reports
  10. Metadata
  11. Statistics
  12. Hashes
  13. Versions
  14. Registry
  15. Charts (optional)
  16. Unit tests

Ends with:
====================================================
PHASE 12 AUTOMATIC MODEL RETRAINING VERIFICATION PASSED
====================================================
"""
import json
import os
import sys
import traceback
from typing import Any, Dict, List

# Add project root to sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

_PASS = "✅ PASS"
_FAIL = "❌ FAIL"
_CONFIG_PATH = "ml/retraining/retraining_config.yaml"


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _ok(msg: str) -> None:
    print(f"  {_PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_FAIL}  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
def verify_config() -> bool:
    _section("1. Configuration Loading")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)

        assert cfg.primary_metric, "primary_metric is empty"
        assert isinstance(cfg.minimum_thresholds, dict), "minimum_thresholds not a dict"
        assert isinstance(cfg.feedback_label_mapping, dict), "label_mapping not a dict"
        assert cfg.get_approved_feedback_path(), "approved_feedback_path not configured"
        assert cfg.get_training_dataset_path(), "training_dataset_path not configured"
        assert cfg.get_output_dir("logs_dir"), "logs_dir not configured"
        assert cfg.get_output_dir("reports_dir"), "reports_dir not configured"
        _ok("Configuration loaded and validated (all required keys present).")
        return True
    except Exception as exc:
        _fail(f"Configuration verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 2. APPROVED FEEDBACK LOADING
# ══════════════════════════════════════════════════════════════════════════════
def verify_approved_feedback_loading() -> bool:
    _section("2. Approved Feedback Loading")
    try:
        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.approved_feedback_loader import ApprovedFeedbackLoader

        cfg = RetrainingConfig(_CONFIG_PATH)

        # Ensure there is at least one record in the approved CSV
        csv_path = cfg.get_approved_feedback_path()
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            _fail(f"Approved feedback CSV not found or empty: {csv_path}")
            return False

        loader = ApprovedFeedbackLoader(cfg)
        ok, err, df = loader.load()
        if not ok:
            _fail(f"Loader returned failure: {err}")
            return False

        assert not df.empty, "Loaded DataFrame is empty"
        assert "label" in df.columns, "'label' column missing after load"
        assert df["label"].isin([0, 1]).all(), "Invalid label values"
        _ok(f"Loaded {len(df)} approved feedback records with valid labels.")
        return True
    except Exception as exc:
        _fail(f"Approved feedback loading failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET MERGE
# ══════════════════════════════════════════════════════════════════════════════
def verify_dataset_merge() -> bool:
    _section("3. Dataset Merge")
    try:
        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.approved_feedback_loader import ApprovedFeedbackLoader
        from ml.retraining.dataset_merge_manager import DatasetMergeManager
        from ml.retraining.dataset_validator import DatasetValidator
        import json

        cfg = RetrainingConfig(_CONFIG_PATH)

        # Load approved feedback
        loader = ApprovedFeedbackLoader(cfg)
        ok, err, approved_df = loader.load()
        if not ok:
            _fail(f"Could not load approved feedback for merge test: {err}")
            return False

        # Load training dataset
        training_csv = cfg.get_training_dataset_path()
        feature_names_path = cfg.get_feature_names_path()

        if not os.path.exists(training_csv):
            _fail(f"Training dataset not found: {training_csv}")
            return False

        existing_df = pd.read_csv(training_csv)
        feature_cols: List[str] = []
        if os.path.exists(feature_names_path):
            with open(feature_names_path) as fh:
                feature_cols = json.load(fh)
        else:
            exclude = {"label", "id"}
            feature_cols = [c for c in existing_df.columns if c not in exclude]

        merger = DatasetMergeManager(cfg)
        ok, err, merged_df = merger.merge(existing_df, approved_df, feature_cols)
        if not ok:
            _fail(f"Merge failed: {err}")
            return False

        assert len(merged_df) >= len(existing_df), "Merged dataset is smaller than original"

        # Validate merged
        validator = DatasetValidator()
        ok, errors = validator.validate(merged_df, context="merged_dataset", check_duplicates=False)
        if not ok:
            _fail(f"Merged dataset validation failed: {errors}")
            return False

        stats = merger.compute_merge_stats(existing_df, approved_df, merged_df)
        assert stats["merged_rows"] == len(merged_df), "Statistics row count mismatch"

        _ok(f"Merged dataset: {len(existing_df)} → {len(merged_df)} rows. Validation passed.")
        return True
    except Exception as exc:
        _fail(f"Dataset merge verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 4. FULL PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def verify_pipeline_execution() -> bool:
    _section("4. Full Retraining Pipeline Execution")
    try:
        from ml.retraining.retraining_pipeline import RetrainingPipeline

        pipeline = RetrainingPipeline(_CONFIG_PATH)
        success = pipeline.run()
        if not success:
            _fail("RetrainingPipeline.run() returned False.")
            return False
        _ok("RetrainingPipeline executed successfully.")
        return True
    except Exception as exc:
        _fail(f"Pipeline execution failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORTS
# ══════════════════════════════════════════════════════════════════════════════
def verify_reports() -> bool:
    _section("5. Reports")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)

        reports = {
            "retraining_report.md": cfg.get_output_file("retraining_report"),
            "comparison_report.md": cfg.get_output_file("comparison_report"),
        }
        for name, path in reports.items():
            if not path or not os.path.isfile(path):
                _fail(f"Report missing: {name} at {path}")
                return False
            if os.path.getsize(path) == 0:
                _fail(f"Report is empty: {name}")
                return False
            with open(path, "r", encoding="utf-8") as fh:
                content = fh.read()
            if "Phase 12" not in content:
                _fail(f"Report does not mention 'Phase 12': {name}")
                return False
            _ok(f"Report verified: {name}")
        return True
    except Exception as exc:
        _fail(f"Reports verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 6. METADATA
# ══════════════════════════════════════════════════════════════════════════════
def verify_metadata() -> bool:
    _section("6. Metadata")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        path = cfg.get_output_file("retraining_metadata")

        if not path or not os.path.isfile(path):
            _fail(f"Metadata file not found: {path}")
            return False

        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        required_keys = ["run_id", "phase", "environment", "library_versions", "run_summary"]
        for key in required_keys:
            if key not in meta:
                _fail(f"Metadata missing required key: '{key}'")
                return False

        assert meta["phase"] == 12, "Metadata phase != 12"
        _ok(f"Metadata verified: run_id={meta.get('run_id')}, phase={meta.get('phase')}")
        return True
    except Exception as exc:
        _fail(f"Metadata verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 7. STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def verify_statistics() -> bool:
    _section("7. Statistics")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        path = cfg.get_output_file("retraining_statistics")

        if not path or not os.path.isfile(path):
            _fail(f"Statistics file not found: {path}")
            return False

        with open(path, "r", encoding="utf-8") as fh:
            stats = json.load(fh)

        required_keys = ["run_id", "dataset", "training", "evaluation", "pipeline"]
        for key in required_keys:
            if key not in stats:
                _fail(f"Statistics missing required key: '{key}'")
                return False

        decision = stats.get("pipeline", {}).get("decision", "")
        assert decision in ("PROMOTED", "REJECTED"), f"Invalid decision in stats: {decision}"
        _ok(f"Statistics verified: decision={decision}")
        return True
    except Exception as exc:
        _fail(f"Statistics verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 8. HASHES
# ══════════════════════════════════════════════════════════════════════════════
def verify_hashes() -> bool:
    _section("8. Hashes")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        path = cfg.get_output_file("retraining_hashes")

        if not path or not os.path.isfile(path):
            _fail(f"Hashes file not found: {path}")
            return False

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert "files" in data, "Hashes JSON missing 'files' key"
        file_hashes = data["files"]
        assert len(file_hashes) > 0, "No hashes recorded"

        _ok(f"Hashes verified: {len(file_hashes)} file checksums recorded.")
        return True
    except Exception as exc:
        _fail(f"Hashes verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 9. VERSIONS
# ══════════════════════════════════════════════════════════════════════════════
def verify_versions() -> bool:
    _section("9. Versions")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        path = cfg.get_output_file("retraining_versions")

        if not path or not os.path.isfile(path):
            _fail(f"Versions file not found: {path}")
            return False

        with open(path, "r", encoding="utf-8") as fh:
            versions = json.load(fh)

        assert isinstance(versions, list), "Versions file is not a list"
        assert len(versions) > 0, "No version entries found"

        latest = versions[-1]
        for key in ("version", "run_id", "decision", "promoted", "file_hashes"):
            assert key in latest, f"Version entry missing key: '{key}'"

        _ok(
            f"Versions verified: {len(versions)} entries, "
            f"latest version={latest['version']}, decision={latest['decision']}."
        )
        return True
    except Exception as exc:
        _fail(f"Versions verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 10. REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
def verify_registry() -> bool:
    _section("10. Model Registry")
    try:
        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.model_registry import RetrainingModelRegistry

        cfg = RetrainingConfig(_CONFIG_PATH)
        registry = RetrainingModelRegistry(cfg.get_registry_file())
        data = registry.load()

        runs = data.get("retraining_runs", [])
        assert len(runs) > 0, "Registry has no retraining runs"

        latest_run = runs[-1]
        for key in ("run_id", "decision", "promoted", "candidate_model_key"):
            assert key in latest_run, f"Registry run entry missing key: '{key}'"

        _ok(f"Registry verified: {len(runs)} run(s), latest decision={latest_run['decision']}.")
        return True
    except Exception as exc:
        _fail(f"Registry verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 11. DEPLOYMENT DECISION
# ══════════════════════════════════════════════════════════════════════════════
def verify_deployment_decision() -> bool:
    _section("11. Deployment Decision")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        path = cfg.get_output_file("deployment_decision")

        if not path or not os.path.isfile(path):
            _fail(f"Deployment decision file not found: {path}")
            return False

        with open(path, "r", encoding="utf-8") as fh:
            decision = json.load(fh)

        required_keys = [
            "retraining_run_id",
            "decision",
            "reason",
            "candidate_model_key",
            "comparison_summary",
        ]
        for key in required_keys:
            if key not in decision:
                _fail(f"Deployment decision missing key: '{key}'")
                return False

        valid_decisions = {"PROMOTED", "REJECTED"}
        dec = decision["decision"]
        assert dec in valid_decisions, f"Invalid decision value: {dec}"

        _ok(f"Deployment decision verified: {dec} (reason: {decision['reason'][:80]}…)")
        return True
    except Exception as exc:
        _fail(f"Deployment decision verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 12. CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def verify_charts() -> bool:
    _section("12. Charts (optional)")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        charts_dir = cfg.get_output_dir("charts_dir")

        if not os.path.isdir(charts_dir):
            _ok("Charts directory not found — skipping (matplotlib may be unavailable).")
            return True

        pngs = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
        if pngs:
            _ok(f"Charts directory contains {len(pngs)} chart(s): {pngs}.")
        else:
            _ok("No charts generated (matplotlib may be unavailable). Continuing.")
        return True
    except Exception as exc:
        _fail(f"Charts verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 13. MODEL COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════
def verify_model_comparator() -> bool:
    _section("13. Model Comparator Logic")
    try:
        from ml.retraining.retraining_config import RetrainingConfig
        from ml.retraining.model_comparator import ModelComparator

        cfg = RetrainingConfig(_CONFIG_PATH)
        comparator = ModelComparator(cfg)

        # Test promotion case
        candidate = {"f1_score": 0.90, "accuracy": 0.90, "precision": 0.90,
                     "recall": 0.90, "roc_auc": 0.95, "mcc": 0.80}
        production = {"f1_score": 0.70, "accuracy": 0.70, "precision": 0.70,
                      "recall": 0.70, "roc_auc": 0.75, "mcc": 0.60}
        result = comparator.compare(candidate, production)
        assert result["promote"] is True, "Expected promotion when candidate is clearly better"

        # Test rejection case
        weak_candidate = {"f1_score": 0.50, "accuracy": 0.50, "precision": 0.50,
                          "recall": 0.50, "roc_auc": 0.55, "mcc": 0.30}
        result2 = comparator.compare(weak_candidate, production)
        assert result2["promote"] is False, "Expected rejection when candidate fails thresholds"

        _ok("Model comparator logic verified (promotion and rejection cases pass).")
        return True
    except Exception as exc:
        _fail(f"Model comparator verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 14. CANDIDATE MODEL ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
def verify_candidate_artifacts() -> bool:
    _section("14. Candidate Model Artifacts")
    try:
        from ml.retraining.retraining_config import RetrainingConfig

        cfg = RetrainingConfig(_CONFIG_PATH)
        candidate_dir = cfg.get_candidate_models_dir()

        if not os.path.isdir(candidate_dir):
            _fail(f"Candidate models directory missing: {candidate_dir}")
            return False

        model_dirs = [
            d for d in os.listdir(candidate_dir)
            if os.path.isdir(os.path.join(candidate_dir, d))
        ]
        if not model_dirs:
            _fail(f"No candidate model sub-directories found in {candidate_dir}")
            return False

        for model_dir in model_dirs:
            model_file = os.path.join(candidate_dir, model_dir, "model.joblib")
            if not os.path.isfile(model_file):
                _fail(f"model.joblib missing for candidate '{model_dir}'")
                return False

        _ok(f"Candidate model artifacts verified: {model_dirs}")
        return True
    except Exception as exc:
        _fail(f"Candidate artifact verification failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 15. UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════
def verify_unit_tests() -> bool:
    _section("15. Unit Tests")

    if "PYTEST_CURRENT_TEST" in os.environ:
        _ok("Active pytest session detected — skipping nested pytest invocation.")
        return True

    try:
        import pytest

        test_path = os.path.join(
            os.path.dirname(__file__),
            "ml", "retraining", "tests", "test_retraining.py"
        )
        if not os.path.isfile(test_path):
            _fail(f"Unit test file not found: {test_path}")
            return False

        print(f"\n  Running pytest on: {test_path}\n")
        # Added "-s" to prevent stdout capture which causes WinError 6 during logging
        exit_code = pytest.main(["-v", "-s", "--tb=short", test_path])
        if exit_code != 0:
            _fail(f"Unit tests failed with exit code {exit_code}.")
            return False
        _ok("All unit tests passed.")
        return True
    except Exception as exc:
        _fail(f"Unit test execution failed: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN VERIFICATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def verify() -> bool:
    """
    Runs all Phase 12 verification checks in order.

    Returns:
        ``True`` when all checks pass, ``False`` otherwise.
    """
    print("\n" + "=" * 60)
    print("  PHASE 12 AUTOMATIC MODEL RETRAINING VERIFICATION")
    print("=" * 60)

    checks = [
        ("Configuration", verify_config),
        ("Approved Feedback Loading", verify_approved_feedback_loading),
        ("Dataset Merge", verify_dataset_merge),
        ("Model Comparator Logic", verify_model_comparator),
        ("Full Pipeline Execution", verify_pipeline_execution),
        ("Reports", verify_reports),
        ("Metadata", verify_metadata),
        ("Statistics", verify_statistics),
        ("Hashes", verify_hashes),
        ("Versions", verify_versions),
        ("Registry", verify_registry),
        ("Deployment Decision", verify_deployment_decision),
        ("Candidate Model Artifacts", verify_candidate_artifacts),
        ("Charts", verify_charts),
        ("Unit Tests", verify_unit_tests),
    ]

    results: Dict[str, bool] = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001
            _fail(f"Unexpected error in '{name}': {exc}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("====================================================")
        print("PHASE 12 AUTOMATIC MODEL RETRAINING VERIFICATION PASSED")
        print("====================================================")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"VERIFICATION FAILED — {len(failed)} check(s) did not pass:")
        for n in failed:
            print(f"  ❌ {n}")

    # Cleanup logger
    try:
        from ml.retraining.retraining_logger import shutdown_logger
        shutdown_logger()
    except Exception:
        pass

    return all_passed


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
