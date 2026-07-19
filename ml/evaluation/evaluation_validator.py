import os
import json
import logging
import pandas as pd
from typing import Tuple, List, Dict, Any
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_utils import compute_file_sha256

logger = logging.getLogger("model_evaluation_pipeline")

class EvaluationValidator:
    """
    Performs integrity validation on all inputs and outputs of the evaluation phase.
    Ensures directories, datasets, loaded models, predictions, metrics, comparison tables,
    leaderboards, reports, statistics, hashes, and versions are complete and valid.
    """
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Executes all validation checks.
        
        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        errors = []
        
        # 1. Validate output directory structure
        logger.info("Validating directory structure...")
        dir_keys = [
            "reports_dir", "statistics_dir", "metadata_dir", "hashes_dir",
            "versions_dir", "leaderboard_dir", "comparison_dir", "logs_dir",
            "predictions_dir", "classification_reports_dir", "confusion_matrices_dir",
            "roc_curves_dir", "precision_recall_curves_dir", "charts_dir"
        ]
        for key in dir_keys:
            path = self.config.get_output_dir(key)
            if not os.path.exists(path):
                errors.append(f"Output directory missing: {path} (key: {key})")

        # 2. Validate input files
        logger.info("Validating input files...")
        input_csv = self.config.get_input_path("dataset_csv")
        if not os.path.exists(input_csv):
            errors.append(f"Input dataset CSV file missing: {input_csv}")
            
        feature_names_json = self.config.get_input_path("feature_names_json")
        if not os.path.exists(feature_names_json):
            errors.append(f"Input feature names JSON file missing: {feature_names_json}")
            
        registry_json = self.config.get_input_path("training_registry_json")
        if not os.path.exists(registry_json):
            errors.append(f"Training registry JSON file missing: {registry_json}")

        # 3. Validate predictions export CSVs
        logger.info("Validating prediction exports...")
        pred_dir = self.config.get_output_dir("predictions_dir")
        enabled_models = ["logistic_regression", "svm", "random_forest", "xgboost"]
        
        for m_key in enabled_models:
            pred_file = os.path.join(pred_dir, f"predictions_{m_key}.csv")
            if not os.path.exists(pred_file):
                errors.append(f"Predictions CSV missing for '{m_key}': {pred_file}")
            else:
                try:
                    df_pred = pd.read_csv(pred_file)
                    expected_cols = ["Actual Label", "Predicted Label", "Prediction Probability", "Correct Prediction"]
                    for col in expected_cols:
                        if col not in df_pred.columns:
                            errors.append(f"Column '{col}' missing from predictions CSV of '{m_key}': {pred_file}")
                except Exception as e:
                    errors.append(f"Failed to read predictions CSV for '{m_key}': {e}")

        # 4. Validate metrics files
        logger.info("Validating classification reports and confusion matrices...")
        conf_dir = self.config.get_output_dir("confusion_matrices_dir")
        class_dir = self.config.get_output_dir("classification_reports_dir")
        
        for m_key in enabled_models:
            # Confusion matrix JSON
            cm_json = os.path.join(conf_dir, f"confusion_matrix_{m_key}.json")
            if not os.path.exists(cm_json):
                errors.append(f"Confusion matrix JSON missing for '{m_key}': {cm_json}")
            else:
                try:
                    with open(cm_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    required_keys = ["true_negative", "false_positive", "false_negative", "true_positive"]
                    for k in required_keys:
                        if k not in data:
                            errors.append(f"Key '{k}' missing from confusion matrix JSON of '{m_key}': {cm_json}")
                except Exception as e:
                    errors.append(f"Failed to parse confusion matrix JSON for '{m_key}': {e}")
                    
            # Classification report JSON
            cr_json = os.path.join(class_dir, f"classification_report_{m_key}.json")
            if not os.path.exists(cr_json):
                errors.append(f"Classification report JSON missing for '{m_key}': {cr_json}")
            else:
                try:
                    with open(cr_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Check macro avg and weighted avg keys
                    for avg in ["macro avg", "weighted avg", "accuracy"]:
                        if avg not in data:
                            errors.append(f"Key '{avg}' missing from classification report JSON of '{m_key}': {cr_json}")
                except Exception as e:
                    errors.append(f"Failed to parse classification report JSON for '{m_key}': {e}")

        # 5. Validate model comparison table
        logger.info("Validating comparison tables...")
        comp_csv = self.config.get_output_path("comparison_csv_file")
        comp_json = self.config.get_output_path("comparison_json_file")
        comp_md = self.config.get_output_path("comparison_md_file")
        
        for p in [comp_csv, comp_json, comp_md]:
            if not os.path.exists(p):
                errors.append(f"Comparison table file missing: {p}")
            elif os.path.getsize(p) == 0:
                errors.append(f"Comparison table file is empty: {p}")

        # 6. Validate leaderboard
        logger.info("Validating leaderboards...")
        lead_csv = self.config.get_output_path("leaderboard_csv_file")
        lead_json = self.config.get_output_path("leaderboard_json_file")
        lead_md = self.config.get_output_path("leaderboard_md_file")
        
        for p in [lead_csv, lead_json, lead_md]:
            if not os.path.exists(p):
                errors.append(f"Leaderboard file missing: {p}")
            elif os.path.getsize(p) == 0:
                errors.append(f"Leaderboard file is empty: {p}")

        # 7. Validate Best Model
        logger.info("Validating best model metadata...")
        best_model_path = self.config.get_output_path("best_model_file")
        if not os.path.exists(best_model_path):
            errors.append(f"Best model JSON file missing: {best_model_path}")
        else:
            try:
                with open(best_model_path, "r", encoding="utf-8") as f:
                    best = json.load(f)
                required_keys = ["model_key", "model_id", "algorithm", "selection_metric_used", "overall_score", "metrics", "path"]
                for k in required_keys:
                    if k not in best:
                        errors.append(f"Key '{k}' missing from best model JSON: {best_model_path}")
            except Exception as e:
                errors.append(f"Failed to parse best model JSON: {e}")

        # 8. Validate Statistics
        logger.info("Validating evaluation statistics...")
        stats_file = self.config.get_output_path("evaluation_statistics_file")
        if not os.path.exists(stats_file):
            errors.append(f"Evaluation statistics file missing: {stats_file}")
        else:
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                required_keys = [
                    "dataset_size_bytes", "dataset_rows", "feature_count", 
                    "prediction_times", "average_prediction_time_sec", 
                    "memory_usages", "total_models_evaluated", "best_model", 
                    "evaluation_pipeline_duration_sec"
                ]
                for k in required_keys:
                    if k not in stats:
                        errors.append(f"Key '{k}' missing from evaluation statistics JSON: {stats_file}")
            except Exception as e:
                errors.append(f"Failed to parse evaluation statistics JSON: {e}")

        # 9. Validate report
        logger.info("Validating report markdown...")
        report_file = self.config.get_output_path("evaluation_report_file")
        if not os.path.exists(report_file):
            errors.append(f"Evaluation report file missing: {report_file}")
        else:
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    text = f.read()
                if len(text.strip()) < 100:
                    errors.append("Evaluation report markdown is too short or empty.")
                if "Phase 7" not in text and "Model Evaluation" not in text:
                    errors.append("Evaluation report does not mention Phase 7 Model Evaluation.")
            except Exception as e:
                errors.append(f"Failed to read evaluation report: {e}")

        # 10. Validate versions file
        logger.info("Validating version records...")
        versions_file = self.config.get_output_path("versions_file")
        if not os.path.exists(versions_file):
            errors.append(f"Versions file missing: {versions_file}")
        else:
            try:
                with open(versions_file, "r", encoding="utf-8") as f:
                    versions = json.load(f)
                if not isinstance(versions, list) or len(versions) == 0:
                    errors.append(f"Versions JSON must be a non-empty list of runs: {versions_file}")
                else:
                    last_run = versions[-1]
                    required_keys = [
                        "version", "timestamp", "evaluation_version", 
                        "training_version", "dataset_version", "feature_selection_version", 
                        "best_model_id", "hash", "files"
                    ]
                    for k in required_keys:
                        if k not in last_run:
                            errors.append(f"Key '{k}' missing from latest run in versions JSON: {versions_file}")
            except Exception as e:
                errors.append(f"Failed to parse versions JSON: {e}")

        # 11. Validate hashes file
        logger.info("Validating hash checklist...")
        hashes_file = self.config.get_output_path("hashes_file")
        if not os.path.exists(hashes_file):
            errors.append(f"Hashes file missing: {hashes_file}")
        else:
            try:
                with open(hashes_file, "r", encoding="utf-8") as f:
                    hashes = json.load(f)
                if not isinstance(hashes, dict) or len(hashes) == 0:
                    errors.append(f"Hashes JSON must be a non-empty dictionary: {hashes_file}")
                else:
                    # Check each registered hash file's actual hash against the registered key
                    for registered_hash, file_path in hashes.items():
                        if os.path.exists(file_path):
                            actual_hash = compute_file_sha256(file_path)
                            if actual_hash == "N/A":
                                errors.append(f"Failed to compute actual hash for {file_path}")
                            elif actual_hash != registered_hash:
                                errors.append(f"Hash discrepancy detected. File '{file_path}' hash changed from {registered_hash} to {actual_hash}")
            except Exception as e:
                errors.append(f"Failed to parse hashes JSON: {e}")

        success = len(errors) == 0
        if success:
            logger.info("Validation complete. All outputs verified successfully.")
        else:
            logger.error(f"Validation failed with {len(errors)} validation errors.")
            
        return success, errors
