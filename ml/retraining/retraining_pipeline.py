"""
retraining_pipeline.py
Orchestrates the complete Phase 12: Automatic Model Retraining System.

Workflow
--------
1.  Load approved feedback (Phase 11 output).
2.  Validate approved records.
3.  Load existing training dataset (Phase 5 output).
4.  Merge approved feedback with training dataset.
5.  Validate merged dataset integrity.
6.  Train candidate models on merged dataset (Phase 6 components).
7.  Evaluate candidate models (Phase 7 components).
8.  Load production model metrics for comparison.
9.  Compare candidate vs. production (acceptance policy).
10. Execute deployment decision (promote / reject).
11. Update model registry.
12. Generate statistics, metadata, hashes, versions.
13. Generate reports (retraining_report.md, comparison_report.md).
14. Generate visualization charts.
15. Validate all artifacts.
"""
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ml.retraining.retraining_config import RetrainingConfig
from ml.retraining.retraining_logger import setup_logger
from ml.retraining.approved_feedback_loader import ApprovedFeedbackLoader
from ml.retraining.dataset_validator import DatasetValidator
from ml.retraining.dataset_merge_manager import DatasetMergeManager
from ml.retraining.training_executor import TrainingExecutor
from ml.retraining.evaluation_executor import EvaluationExecutor
from ml.retraining.model_comparator import ModelComparator
from ml.retraining.deployment_manager import DeploymentManager
from ml.retraining.model_registry import RetrainingModelRegistry
from ml.retraining.statistics_manager import RetrainingStatisticsManager
from ml.retraining.metadata_manager import RetrainingMetadataManager
from ml.retraining.report_generator import RetrainingReportGenerator
from ml.retraining.visualization import RetrainingVisualizer
from ml.retraining.hash_generator import HashGenerator
from ml.retraining.version_manager import RetrainingVersionManager
from ml.retraining.validator import RetrainingArtifactValidator


class RetrainingPipeline:
    """
    Orchestrates the full Phase 12 Automatic Model Retraining lifecycle.

    Usage
    -----
    .. code-block:: python

        pipeline = RetrainingPipeline()
        success = pipeline.run()
    """

    def __init__(
        self,
        config_path: str = "ml/retraining/retraining_config.yaml",
    ) -> None:
        self.config = RetrainingConfig(config_path)
        self.logger = setup_logger(self.config.get_output_dir("logs_dir"))
        self._run_id: str = f"retrain_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # ── Public Entry Point ────────────────────────────────────────────────────────────────────────
    def run(self) -> bool:
        """
        Executes the complete retraining pipeline.

        Returns:
            ``True`` on success, ``False`` on any unrecoverable error.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 12 AUTOMATIC MODEL RETRAINING PIPELINE")
        self.logger.info("Run ID: %s", self._run_id)
        self.logger.info("=" * 80)

        pipeline_start = time.perf_counter()
        saved_files: Dict[str, str] = {}

        try:
            # ── Step 1: Load Approved Feedback ────────────────────────────────────────────────────
            self.logger.info("--- Step 1: Loading Approved Feedback ---")
            self._ensure_output_dirs()

            loader = ApprovedFeedbackLoader(self.config)
            ok, err, approved_df = loader.load()
            if not ok:
                self.logger.error("Step 1 FAILED: %s", err)
                return False
            self.logger.info("Loaded %d approved feedback records.", len(approved_df))

            # ── Step 2: Load Existing Training Dataset ────────────────────────────────────────────
            self.logger.info("--- Step 2: Loading Existing Training Dataset ---")
            training_csv = self.config.get_training_dataset_path()
            feature_names_json = self.config.get_feature_names_path()

            existing_df, feature_columns = self._load_training_dataset(
                training_csv, feature_names_json
            )
            if existing_df is None:
                self.logger.error(
                    "Step 2 FAILED: Could not load training dataset from %s", training_csv
                )
                return False
            self.logger.info(
                "Loaded training dataset: %d rows, %d features.",
                len(existing_df),
                len(feature_columns),
            )

            # ── Step 3: Validate Approved Dataset ─────────────────────────────────────────────────
            self.logger.info("--- Step 3: Validating Approved Feedback ---")
            validator = DatasetValidator()
            ok, errors = validator.validate(
                approved_df,
                context="approved_feedback",
                required_columns=["Feedback ID", "label"],
                check_duplicates=True,
            )
            if not ok:
                self.logger.error("Step 3 FAILED – validation errors: %s", errors)
                return False

            # ── Step 4: Merge Datasets ────────────────────────────────────────────────────────────
            self.logger.info("--- Step 4: Merging Approved Feedback with Training Dataset ---")
            merger = DatasetMergeManager(self.config)
            ok, err, merged_df = merger.merge(existing_df, approved_df, feature_columns)
            if not ok:
                self.logger.error("Step 4 FAILED: %s", err)
                return False

            merge_stats = merger.compute_merge_stats(existing_df, approved_df, merged_df)
            self.logger.info(
                "Merge complete: %d → %d rows.", len(existing_df), len(merged_df)
            )

            # ── Step 5: Validate Merged Dataset Integrity ─────────────────────────────────────────
            self.logger.info("--- Step 5: Validating Merged Dataset Integrity ---")
            ok, errors = validator.validate(
                merged_df, context="merged_dataset", check_duplicates=False
            )
            if not ok:
                self.logger.error("Step 5 FAILED – integrity errors: %s", errors)
                return False

            ok, errors = validator.validate_schema_match(
                existing_df, merged_df, context="schema-match"
            )
            if not ok:
                self.logger.error("Step 5 FAILED – schema mismatch: %s", errors)
                return False

            # ── Step 6: Train Candidate Models ────────────────────────────────────────────────────
            self.logger.info("--- Step 6: Training Candidate Models ---")
            trainer = TrainingExecutor(self.config)
            ok, err, training_summary = trainer.execute(merged_df, feature_columns)
            if not ok:
                self.logger.error("Step 6 FAILED: %s", err)
                return False

            # ── Step 7: Evaluate Candidate Models ─────────────────────────────────────────────────
            self.logger.info("--- Step 7: Evaluating Candidate Models ---")
            X_test: pd.DataFrame = training_summary["X_test"]
            y_test: pd.Series = training_summary["y_test"]
            candidate_dir = self.config.get_candidate_models_dir()
            models_trained: List[str] = training_summary.get("models_trained", [])

            # Normalise model keys for directory lookup
            model_keys: List[str] = [
                self._algorithm_to_key(m) for m in models_trained
            ]

            evaluator = EvaluationExecutor(candidate_dir)
            ok, err, candidate_results = evaluator.execute(X_test, y_test, model_keys)
            if not ok:
                self.logger.error("Step 7 FAILED: %s", err)
                return False

            # Pick best candidate by primary metric
            primary = self.config.primary_metric
            best_key, best_metrics = self._pick_best_candidate(
                candidate_results, primary
            )
            self.logger.info(
                "Best candidate: '%s' (%s=%.4f)",
                best_key,
                primary,
                float(best_metrics.get(primary, 0.0)),
            )

            # ── Step 8: Load Production Metrics ───────────────────────────────────────────────────
            self.logger.info("--- Step 8: Loading Production Model Metrics ---")
            production_metrics = self._load_production_metrics()

            # ── Step 9: Compare Candidate vs. Production ──────────────────────────────────────────
            self.logger.info("--- Step 9: Comparing Candidate vs. Production ---")
            comparator = ModelComparator(self.config)
            comparison_result = comparator.compare(
                candidate_metrics=best_metrics,
                production_metrics=production_metrics,
                candidate_model_key=best_key,
                production_model_key="production",
            )
            promote: bool = comparison_result["promote"]
            decision_str: str = "PROMOTED" if promote else "REJECTED"

            # ── Step 10: Deployment Decision ──────────────────────────────────────────────────────
            self.logger.info("--- Step 10: Executing Deployment Decision (%s) ---", decision_str)
            deployer = DeploymentManager(self.config)
            deployment_decision = deployer.execute(
                comparison_result=comparison_result,
                candidate_model_key=best_key,
                candidate_metrics=best_metrics,
                production_metrics=production_metrics,
                retraining_run_id=self._run_id,
            )
            saved_files["deployment_decision"] = self.config.get_output_file("deployment_decision")

            # ── Step 11: Update Model Registry ────────────────────────────────────────────────────
            self.logger.info("--- Step 11: Updating Model Registry ---")
            registry = RetrainingModelRegistry(self.config.get_registry_file())
            registry.register_run(
                run_id=self._run_id,
                decision=decision_str,
                candidate_model_key=best_key,
                candidate_metrics=best_metrics,
                production_metrics=production_metrics,
                approved_records=len(approved_df),
                merged_rows=len(merged_df),
                promoted=promote,
                reason=comparison_result["reason"],
            )
            saved_files["registry"] = self.config.get_registry_file()

            # ── Step 12: Generate Statistics ──────────────────────────────────────────────────────
            self.logger.info("--- Step 12: Generating Retraining Statistics ---")
            pipeline_duration = time.perf_counter() - pipeline_start
            stats_mgr = RetrainingStatisticsManager()
            stats_mgr.generate(
                run_id=self._run_id,
                approved_records=len(approved_df),
                existing_rows=len(existing_df),
                merged_rows=len(merged_df),
                feature_count=len(feature_columns),
                training_rows=training_summary.get("training_rows", 0),
                testing_rows=training_summary.get("testing_rows", 0),
                pipeline_duration_sec=pipeline_duration,
                candidate_metrics=best_metrics,
                production_metrics=production_metrics,
                decision=decision_str,
                model_durations=training_summary.get("model_durations", {}),
            )
            stats_path = self.config.get_output_file("retraining_statistics")
            stats_mgr.save(stats_path)
            saved_files["retraining_statistics"] = stats_path

            # ── Step 13: Generate Metadata ─────────────────────────────────────────────────────────
            self.logger.info("--- Step 13: Generating Retraining Metadata ---")
            meta_mgr = RetrainingMetadataManager(
                self.config.get_output_dir("metadata_dir")
            )
            meta_mgr.generate_and_save(
                run_id=self._run_id,
                config_path=self.config.config_path,
                approved_csv_path=loader.csv_path,
                approved_records=len(approved_df),
                merged_rows=len(merged_df),
                feature_count=len(feature_columns),
                candidate_model_key=best_key,
                decision=decision_str,
                pipeline_duration_sec=pipeline_duration,
            )
            saved_files["retraining_metadata"] = self.config.get_output_file("retraining_metadata")

            # ── Step 14: Generate Reports ──────────────────────────────────────────────────────────
            self.logger.info("--- Step 14: Generating Reports ---")
            report_gen = RetrainingReportGenerator(
                self.config.get_output_dir("reports_dir")
            )
            retrain_report_path = report_gen.generate_retraining_report(
                run_id=self._run_id,
                approved_records=len(approved_df),
                merge_stats=merge_stats,
                training_summary=training_summary,
                candidate_results=candidate_results,
                decision=decision_str,
                reason=comparison_result["reason"],
                pipeline_duration_sec=pipeline_duration,
                generated_files=saved_files,
            )
            saved_files["retraining_report"] = retrain_report_path

            comparison_report_path = report_gen.generate_comparison_report(
                run_id=self._run_id,
                comparison_result=comparison_result,
                candidate_metrics=best_metrics,
                production_metrics=production_metrics,
            )
            saved_files["comparison_report"] = comparison_report_path

            # ── Step 15: Generate Visualization Charts ─────────────────────────────────────────────
            self.logger.info("--- Step 15: Generating Visualization Charts ---")
            visualizer = RetrainingVisualizer(self.config.get_output_dir("charts_dir"))
            chart_paths = visualizer.generate_all(
                candidate_metrics=best_metrics,
                production_metrics=production_metrics,
                model_durations=training_summary.get("model_durations", {}),
                label_distribution=merge_stats.get("label_distribution_merged", {}),
            )
            saved_files.update(
                {f"chart_{name}": path for name, path in chart_paths.items()}
            )

            # ── Step 16: Generate Hashes ───────────────────────────────────────────────────────────
            self.logger.info("--- Step 16: Generating SHA-256 Hashes ---")
            hash_gen = HashGenerator(self.config.get_output_dir("hashes_dir"))
            hashes_path = hash_gen.generate(
                files={k: v for k, v in saved_files.items() if os.path.isfile(str(v))},
                output_filename="retraining_hashes.json",
            )
            saved_files["retraining_hashes"] = hashes_path

            # ── Step 17: Register Version ──────────────────────────────────────────────────────────
            self.logger.info("--- Step 17: Registering Retraining Version ---")
            from ml.retraining.hash_generator import compute_sha256

            file_hashes = {
                k: compute_sha256(v)
                for k, v in saved_files.items()
                if isinstance(v, str) and os.path.isfile(v)
            }
            version_mgr = RetrainingVersionManager(
                self.config.get_output_dir("versions_dir")
            )
            version_mgr.register(
                run_id=self._run_id,
                decision=decision_str,
                promoted=promote,
                file_hashes=file_hashes,
            )
            saved_files["retraining_versions"] = version_mgr.versions_file

            # ── Step 18: Validate Artifacts ────────────────────────────────────────────────────────
            self.logger.info("--- Step 18: Validating All Artifacts ---")
            artifact_validator = RetrainingArtifactValidator(self.config)
            is_valid, validation_errors = artifact_validator.validate_all()
            if not is_valid:
                self.logger.error(
                    "Artifact validation FAILED with %d errors:", len(validation_errors)
                )
                for err in validation_errors:
                    self.logger.error("  - %s", err)
                return False

            self.logger.info("=" * 80)
            self.logger.info(
                "PHASE 12 RETRAINING PIPELINE COMPLETED SUCCESSFULLY "
                "(decision=%s, run_id=%s).",
                decision_str,
                self._run_id,
            )
            self.logger.info("=" * 80)
            return True

        except Exception as exc:  # noqa: BLE001
            self.logger.exception(
                "Unhandled exception in Phase 12 Retraining Pipeline: %s", exc
            )
            raise

    # ── Helpers ───────────────────────────────────────────────────────────────────────────────────
    def _ensure_output_dirs(self) -> None:
        """Creates all configured output directories."""
        dirs = [
            self.config.get_output_dir("logs_dir"),
            self.config.get_output_dir("reports_dir"),
            self.config.get_output_dir("metadata_dir"),
            self.config.get_output_dir("statistics_dir"),
            self.config.get_output_dir("versions_dir"),
            self.config.get_output_dir("hashes_dir"),
            self.config.get_output_dir("charts_dir"),
            self.config.get_output_dir("registry_dir"),
            self.config.get_candidate_models_dir(),
            self.config.get_production_models_dir(),
        ]
        for d in dirs:
            if d:
                os.makedirs(d, exist_ok=True)

    def _load_training_dataset(
        self, csv_path: str, feature_names_path: str
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Loads the Phase 5 feature-selected training dataset."""
        if not os.path.exists(csv_path):
            self.logger.error("Training dataset not found: %s", csv_path)
            return None, []

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Failed to read training dataset: %s", exc)
            return None, []

        # Load feature names
        feature_columns: List[str] = []
        if os.path.exists(feature_names_path):
            try:
                with open(feature_names_path, "r", encoding="utf-8") as fh:
                    feature_columns = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Could not load feature names from %s: %s", feature_names_path, exc
                )

        if not feature_columns:
            # Derive feature columns from DataFrame (exclude label/id)
            exclude = {"label", "id"}
            feature_columns = [c for c in df.columns if c not in exclude]

        return df, feature_columns

    def _load_production_metrics(self) -> Dict[str, Any]:
        """
        Loads the current production model metrics from:
        1. Retraining registry (previous promotion), or
        2. Phase 7 best_model.json (initial production baseline).
        """
        # 1. Try retraining registry
        registry = RetrainingModelRegistry(self.config.get_registry_file())
        prod_metrics = registry.get_production_metrics()
        if prod_metrics:
            self.logger.info("Production metrics loaded from retraining registry.")
            return prod_metrics

        # 2. Fall back to Phase 7 best_model.json
        best_model_path = "ml/evaluation/best_model.json"
        if os.path.exists(best_model_path):
            try:
                with open(best_model_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                metrics = data.get("metrics", {})
                self.logger.info(
                    "Production metrics loaded from Phase 7 best_model.json."
                )
                return metrics
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Could not load best_model.json: %s — using empty baseline.",
                    exc,
                )

        # 3. Return empty baseline (candidate will always win vs. zeros)
        self.logger.warning(
            "No production metrics found — using zero baseline. "
            "Candidate will be promoted if it meets thresholds."
        )
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "mcc": 0.0,
            "balanced_accuracy": 0.0,
            "inference_time_sec": 9999.0,
            "memory_used_mb": 0.0,
        }

    def _pick_best_candidate(
        self,
        candidate_results: Dict[str, Dict[str, Any]],
        primary_metric: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """Selects the best candidate model by *primary_metric*."""
        best_key = ""
        best_val = -1.0
        best_metrics: Dict[str, Any] = {}

        for key, result in candidate_results.items():
            metrics = result.get("metrics", {})
            val = float(metrics.get(primary_metric, 0.0) or 0.0)
            if val > best_val:
                best_val = val
                best_key = key
                best_metrics = metrics

        return best_key, best_metrics

    @staticmethod
    def _algorithm_to_key(algorithm_name: str) -> str:
        """Maps algorithm display name to internal model key."""
        mapping: Dict[str, str] = {
            "Logistic Regression": "logistic_regression",
            "Linear SVM": "svm",
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
        }
        return mapping.get(algorithm_name, algorithm_name.lower().replace(" ", "_"))


# ── Script Entry Point ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pipeline = RetrainingPipeline()
    success = pipeline.run()
    sys.exit(0 if success else 1)
