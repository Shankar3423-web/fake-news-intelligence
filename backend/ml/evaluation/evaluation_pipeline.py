import os
import sys
import time
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple

# Ensure project root is in path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_logger import setup_logger
from ml.evaluation.evaluation_utils import get_memory_usage, compute_file_sha256
from ml.evaluation.dataset_loader import DatasetLoader
from ml.evaluation.evaluator_factory import EvaluatorFactory
from ml.evaluation.confusion_matrix_generator import ConfusionMatrixGenerator
from ml.evaluation.roc_auc_generator import RocAucGenerator
from ml.evaluation.classification_report_generator import ClassificationReportGenerator
from ml.evaluation.comparison_generator import ComparisonGenerator
from ml.evaluation.leaderboard_generator import LeaderboardGenerator
from ml.evaluation.best_model_selector import BestModelSelector
from ml.evaluation.metadata_manager import MetadataManager
from ml.evaluation.hash_generator import HashGenerator
from ml.evaluation.version_manager import VersionManager
from ml.evaluation.report_generator import ReportGenerator
from ml.evaluation.evaluation_validator import EvaluationValidator
from ml.evaluation.evaluation_statistics import EvaluationStatistics
from ml.evaluation.evaluation_profiler import EvaluationProfiler

class EvaluationPipeline:
    """
    Orchestrates the entire Phase 7: Model Evaluation Pipeline.
    Loads test dataset, evaluates Logistic Regression, Linear SVM, Random Forest, XGBoost,
    calculates metrics, exports predictions, generates comparison tables/plots, ranks in leaderboard,
    selects best model, updates metadata, statistics, versions, hashes, and runs validation.
    """
    def __init__(self, config_path: str = "ml/evaluation/evaluation_config.yaml") -> None:
        self.config = EvaluationConfig(config_path)
        self.logger = setup_logger(self.config.get_output_dir("logs_dir"))

    def run(self) -> bool:
        """
        Executes the entire evaluation pipeline.
        
        Returns:
            bool: True if completed successfully without validation failures, False otherwise.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 7 MODEL EVALUATION PIPELINE")
        self.logger.info("=" * 80)
        
        pipeline_start_time = time.perf_counter()
        start_rss, start_peak = get_memory_usage()
        
        warnings: List[str] = []
        model_results: Dict[str, Dict[str, Any]] = {}
        saved_files: Dict[str, str] = {}
        
        try:
            # 1. Load Dataset & Validate
            self.logger.info("--- Step 1: Loading and Validating Test Dataset ---")
            dataset_csv = self.config.get_input_path("dataset_csv")
            feature_names_json = self.config.get_input_path("feature_names_json")
            
            loader = DatasetLoader(dataset_csv, feature_names_json)
            X_test, y_test = loader.load_test_split(self.config)
            
            # Check for label class imbalance
            test_counts = y_test.value_counts().to_dict()
            test_ratio_fake = test_counts.get(1, 0) / len(y_test)
            if abs(test_ratio_fake - 0.5) > 0.15:
                warn_msg = f"Testing class distribution shows imbalance (Fake ratio: {test_ratio_fake:.2%})."
                self.logger.warning(warn_msg)
                warnings.append(warn_msg)

            # 2. Run Predictions & Metrics Calculation for every model
            self.logger.info("--- Step 2: Running Evaluation on Models ---")
            models_to_evaluate = ["logistic_regression", "svm", "random_forest", "xgboost"]
            
            for m_key in models_to_evaluate:
                self.logger.info(f"Evaluating model classifier: '{m_key}'")
                
                with EvaluationProfiler.profile(f"{m_key.upper()} Evaluation") as prof:
                    evaluator = EvaluatorFactory.create_evaluator(m_key, self.config)
                    res = evaluator.evaluate(X_test, y_test)
                    
                model_results[m_key] = res
                
                # Check for specific warnings
                if m_key == "svm" and res["predictions"]["y_prob_type"] == "decision_function":
                    warn_msg = "Linear SVM does not natively support probability estimates. Used decision_function instead."
                    self.logger.info(warn_msg)
                    warnings.append(warn_msg)

            # 3. Export Predictions CSV
            self.logger.info("--- Step 3: Exporting Predictions ---")
            pred_dir = self.config.get_output_dir("predictions_dir")
            os.makedirs(pred_dir, exist_ok=True)
            
            for m_key, res in model_results.items():
                pred_out = res["predictions"]
                df_pred = pd.DataFrame({
                    "Actual Label": y_test.values,
                    "Predicted Label": pred_out["y_pred"],
                    "Prediction Probability": pred_out["y_prob"] if pred_out["y_prob"] is not None else float('nan'),
                    "Correct Prediction": y_test.values == pred_out["y_pred"]
                })
                
                pred_csv_path = os.path.join(pred_dir, f"predictions_{m_key}.csv")
                df_pred.to_csv(pred_csv_path, index=False)
                saved_files[f"predictions_{m_key}"] = pred_csv_path
                self.logger.info(f"Exported prediction outputs for '{m_key}' to {pred_csv_path}")

            # 4. Generate Confusion Matrices
            self.logger.info("--- Step 4: Generating Confusion Matrices ---")
            cm_gen = ConfusionMatrixGenerator(self.config.get_output_dir("confusion_matrices_dir"))
            for m_key, res in model_results.items():
                cm_out = cm_gen.generate(m_key, y_test.values, res["predictions"]["y_pred"])
                saved_files[f"{m_key}_confusion_matrix_json"] = cm_out["json_path"]
                saved_files[f"{m_key}_confusion_matrix_csv"] = cm_out["csv_path"]
                saved_files[f"{m_key}_confusion_matrix_png"] = cm_out["png_path"]

            # 5. Generate ROC Curves
            self.logger.info("--- Step 5: Generating ROC Curves ---")
            if self.config.enable_roc:
                roc_gen = RocAucGenerator(self.config.get_output_dir("roc_curves_dir"))
                for m_key, res in model_results.items():
                    roc_out = roc_gen.generate(m_key, y_test.values, res["predictions"]["y_prob"])
                    if roc_out:
                        saved_files[f"{m_key}_roc_curve_json"] = roc_out["json_path"]
                        saved_files[f"{m_key}_roc_curve_png"] = roc_out["png_path"]

            # 6. Generate Precision-Recall Curves & Classification Reports
            self.logger.info("--- Step 6: Generating PR Curves and Classification Reports ---")
            cr_gen = ClassificationReportGenerator(
                reports_dir=self.config.get_output_dir("classification_reports_dir"),
                pr_curves_dir=self.config.get_output_dir("precision_recall_curves_dir")
            )
            for m_key, res in model_results.items():
                # Classification reports
                cr_out = cr_gen.generate_report(m_key, y_test.values, res["predictions"]["y_pred"])
                saved_files[f"{m_key}_classification_report_json"] = cr_out["json_path"]
                saved_files[f"{m_key}_classification_report_md"] = cr_out["md_path"]
                
                # PR curves
                if self.config.enable_pr_curve:
                    pr_out = cr_gen.generate_pr_curve(m_key, y_test.values, res["predictions"]["y_prob"])
                    if pr_out:
                        saved_files[f"{m_key}_pr_curve_json"] = pr_out["json_path"]
                        saved_files[f"{m_key}_pr_curve_png"] = pr_out["png_path"]

            # 7. Metadata Centralization
            self.logger.info("--- Step 7: Archiving Evaluation Metadata ---")
            meta_manager = MetadataManager(self.config)
            training_ver, dataset_ver, selection_ver = meta_manager.get_latest_training_info()
            
            for m_key, res in model_results.items():
                eval_meta = {
                    "model_key": m_key,
                    "model_id": res["model_id"],
                    "algorithm": res["algorithm"],
                    "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "training_version": training_ver,
                    "dataset_version": dataset_ver,
                    "feature_selection_version": selection_ver,
                    "metrics": res["metrics"]
                }
                meta_path = meta_manager.save_metadata(m_key, eval_meta)
                saved_files[f"{m_key}_central_evaluation_metadata"] = meta_path

            # 8. Best Model Selection & Leaderboard
            self.logger.info("--- Step 8: Selecting Best Model and Ranking Leaderboard ---")
            selector = BestModelSelector(self.config)
            overall_scores = selector.calculate_overall_scores(model_results)
            best_model_key, best_model_data = selector.select_best_model(model_results, overall_scores)
            saved_files["best_model_json"] = self.config.get_output_path("best_model_file")
            
            # Leaderboards
            lead_gen = LeaderboardGenerator(self.config.get_output_dir("leaderboard_dir"))
            leaderboard_list = lead_gen.generate(model_results, overall_scores)
            saved_files["leaderboard_csv"] = self.config.get_output_path("leaderboard_csv_file")
            saved_files["leaderboard_json"] = self.config.get_output_path("leaderboard_json_file")
            saved_files["leaderboard_md"] = self.config.get_output_path("leaderboard_md_file")

            # 9. Model Comparison Matrix
            self.logger.info("--- Step 9: Compiling Comparison Matrices ---")
            comp_gen = ComparisonGenerator(
                comparison_dir=self.config.get_output_dir("comparison_dir"),
                charts_dir=self.config.get_output_dir("charts_dir")
            )
            comp_out = comp_gen.generate(model_results, overall_scores)
            saved_files["comparison_csv"] = comp_out["csv_path"]
            saved_files["comparison_json"] = comp_out["json_path"]
            saved_files["comparison_md"] = comp_out["md_path"]
            
            # Add chart locations to saved files
            saved_files["chart_metrics_comparison"] = os.path.join(self.config.get_output_dir("charts_dir"), "metrics_comparison.png")
            saved_files["chart_prediction_time"] = os.path.join(self.config.get_output_dir("charts_dir"), "prediction_time_comparison.png")
            saved_files["chart_model_size"] = os.path.join(self.config.get_output_dir("charts_dir"), "model_size_comparison.png")

            # 10. Compile Pipeline Statistics
            self.logger.info("--- Step 10: Generating Evaluation Statistics ---")
            pipeline_end_time = time.perf_counter()
            total_duration = pipeline_end_time - pipeline_start_time
            
            stats_manager = EvaluationStatistics()
            stats_manager.generate(
                dataset_path=dataset_csv,
                feature_count=X_test.shape[1],
                dataset_size_rows=len(X_test),
                pipeline_duration=total_duration,
                best_model_key=best_model_key,
                model_results=model_results
            )
            stats_path = self.config.get_output_path("evaluation_statistics_file")
            stats_manager.save(stats_path)
            saved_files["statistics_file"] = stats_path

            # 11. Compile Evaluation Report MD
            self.logger.info("--- Step 11: Compiling Markdown Report ---")
            dataset_details = {
                "path": dataset_csv,
                "size_bytes": os.path.getsize(dataset_csv) if os.path.exists(dataset_csv) else 0,
                "total_rows": len(X_test) * 5, # Approximate original size before 20% split
                "test_rows": len(X_test),
                "test_size": self.config.test_size,
                "feature_count": X_test.shape[1],
                "random_state": self.config.random_state
            }
            training_details = {
                "training_version": training_ver,
                "dataset_version": dataset_ver,
                "feature_selection_version": selection_ver
            }
            
            report_path = self.config.get_output_path("evaluation_report_file")
            report_gen = ReportGenerator(report_path)
            report_gen.generate(
                dataset_info=dataset_details,
                training_info=training_details,
                model_results=model_results,
                overall_scores=overall_scores,
                leaderboard=leaderboard_list,
                best_model=best_model_data,
                pipeline_duration=total_duration,
                generated_files=saved_files,
                warnings=warnings
            )
            saved_files["report_file"] = report_path

            # 12. Hashing deliverables
            self.logger.info("--- Step 12: Generating Checksum Hashes ---")
            hash_gen = HashGenerator(self.config)
            hashes_path = hash_gen.generate_hashes(saved_files)
            saved_files["hashes_file"] = hashes_path
            
            pipeline_digest = compute_file_sha256(hashes_path)

            # 13. Register version runs
            self.logger.info("--- Step 13: Logging Evaluation Versions ---")
            version_manager = VersionManager(self.config)
            version_manager.register_run(
                training_version=training_ver,
                dataset_version=dataset_ver,
                feature_selection_version=selection_ver,
                best_model_id=best_model_data["model_id"],
                pipeline_hash=pipeline_digest,
                files_dict=saved_files
            )
            saved_files["versions_file"] = self.config.get_output_path("versions_file")

            # 14. Validator Run
            self.logger.info("--- Step 14: Running Integrity Validation ---")
            validator = EvaluationValidator(self.config)
            valid, validation_errors = validator.validate_all()
            
            if not valid:
                self.logger.error("Pipeline validation failed with errors:")
                for err in validation_errors:
                    self.logger.error(f"- {err}")
                return False

            self.logger.info("=" * 80)
            self.logger.info("PHASE 7 MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            return True

        except Exception as e:
            self.logger.exception(f"Unhandled exception in Phase 7 Model Evaluation Pipeline: {e}")
            raise

if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    pipeline.run()
