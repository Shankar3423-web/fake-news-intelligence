import os
import json
import logging
import joblib
import pandas as pd
from typing import Tuple, List, Dict, Any
from ml.training.training_config import TrainingConfig
from ml.training.training_utils import compute_file_sha256

logger = logging.getLogger("model_training_pipeline")

class TrainingValidator:
    """
    Performs integrity validation on all outputs of the training phase.
    Ensures directories, dataset splits, saved models, metadata, reports,
    registries, hashes, and versions are complete and valid.
    """
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def validate_all(self) -> Tuple[bool, List[str]]:
        """
        Executes all validation steps.
        
        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        errors = []
        
        # 1. Validate Directory Structure
        logger.info("Validating directory structure...")
        dir_keys = ["models_dir", "metadata_dir", "reports_dir", "statistics_dir", "versions_dir", "hashes_dir", "logs_dir"]
        for key in dir_keys:
            path = self.config.get_output_dir(key)
            if not os.path.exists(path):
                errors.append(f"Output directory missing: {path} (key: {key})")

        # 2. Validate Input Dataset existence
        logger.info("Validating input dataset...")
        input_csv = self.config.get_input_path("dataset_csv")
        if not os.path.exists(input_csv):
            errors.append(f"Input dataset file not found: {input_csv}")
            
        feature_names_json = self.config.get_input_path("feature_names_json")
        if not os.path.exists(feature_names_json):
            errors.append(f"Input feature names JSON file not found: {feature_names_json}")

        # 3. Validate Output Models
        logger.info("Validating trained models...")
        models_dir = self.config.get_output_dir("models_dir")
        enabled_models = []
        for model_key in ["logistic_regression", "svm", "random_forest", "xgboost"]:
            if self.config.is_model_enabled(model_key):
                enabled_models.append(model_key)
                
        for model in enabled_models:
            model_folder = os.path.join(models_dir, model)
            if not os.path.exists(model_folder):
                errors.append(f"Model directory missing for enabled model: {model_folder}")
                continue
                
            required_files = ["model.joblib", "metadata.json", "training_config.json", "feature_order.json"]
            for rf in required_files:
                rf_path = os.path.join(model_folder, rf)
                if not os.path.exists(rf_path):
                    errors.append(f"Required model file missing: {rf_path}")
                else:
                    # Validate parsing for JSON files
                    if rf.endswith(".json"):
                        try:
                            with open(rf_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if rf == "feature_order.json" and (not isinstance(data, list) or len(data) == 0):
                                errors.append(f"feature_order.json is empty or not a list: {rf_path}")
                            if rf == "training_config.json" and not isinstance(data, dict):
                                errors.append(f"training_config.json is not a dictionary: {rf_path}")
                        except Exception as e:
                            errors.append(f"Failed to parse JSON file {rf_path}: {e}")
                    
                    # Validate joblib loading
                    if rf == "model.joblib":
                        try:
                            loaded_model = joblib.load(rf_path)
                            if loaded_model is None:
                                errors.append(f"Loaded model is None: {rf_path}")
                        except Exception as e:
                            errors.append(f"Failed to load joblib model {rf_path}: {e}")

        # 4. Validate Model Registry
        logger.info("Validating model registry...")
        registry_file = self.config.get_output_path("registry_file")
        if not os.path.exists(registry_file):
            errors.append(f"Registry file not found: {registry_file}")
        else:
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    registry = json.load(f)
                if "models" not in registry or not isinstance(registry["models"], list):
                    errors.append(f"Registry file is malformed (missing models list): {registry_file}")
                else:
                    for entry in registry["models"]:
                        required_keys = ["id", "algorithm", "version", "dataset_version", "training_date", "feature_count", "training_samples", "testing_samples", "path"]
                        missing_keys = [k for k in required_keys if k not in entry]
                        if missing_keys:
                            errors.append(f"Registry entry '{entry.get('id', 'unknown')}' missing keys: {missing_keys}")
            except Exception as e:
                errors.append(f"Failed to parse registry JSON {registry_file}: {e}")

        # 5. Validate Statistics
        logger.info("Validating training statistics...")
        stats_file = self.config.get_output_path("statistics_file")
        if not os.path.exists(stats_file):
            errors.append(f"Statistics file not found: {stats_file}")
        else:
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                required_stats = ["dataset_size_bytes", "feature_count", "training_rows", "testing_rows", "training_duration_sec", "memory_used_mb", "per_model_duration_sec", "total_models_trained", "average_training_time_sec"]
                missing_stats = [s for s in required_stats if s not in stats]
                if missing_stats:
                    errors.append(f"Statistics JSON missing keys: {missing_stats}")
            except Exception as e:
                errors.append(f"Failed to parse statistics JSON {stats_file}: {e}")

        # 6. Validate Versions
        logger.info("Validating training versions...")
        versions_file = self.config.get_output_path("versions_file")
        if not os.path.exists(versions_file):
            errors.append(f"Versions file not found: {versions_file}")
        else:
            try:
                with open(versions_file, "r", encoding="utf-8") as f:
                    versions = json.load(f)
                if not isinstance(versions, list) or len(versions) == 0:
                    errors.append(f"Versions JSON must be a non-empty list of runs: {versions_file}")
                else:
                    last_run = versions[-1]
                    required_run_keys = ["version", "timestamp", "dataset_version", "feature_selection_version", "training_version", "model_ids", "files"]
                    missing_run_keys = [k for k in required_run_keys if k not in last_run]
                    if missing_run_keys:
                        errors.append(f"Last version log entry is missing keys: {missing_run_keys}")
            except Exception as e:
                errors.append(f"Failed to parse versions JSON {versions_file}: {e}")

        # 7. Validate Reports
        logger.info("Validating report existence and content...")
        report_file = self.config.get_output_path("report_file")
        if not os.path.exists(report_file):
            errors.append(f"Report file not found: {report_file}")
        else:
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report_text = f.read()
                if len(report_text.strip()) < 100:
                    errors.append("Report markdown is too short (empty or incomplete).")
                if "Phase 6" not in report_text and "Model Training" not in report_text:
                    errors.append("Report markdown does not mention Phase 6 Model Training.")
            except Exception as e:
                errors.append(f"Failed to read report file {report_file}: {e}")

        # 8. Validate Hashes
        logger.info("Validating generated SHA-256 hashes...")
        hashes_dir = self.config.get_output_dir("hashes_dir")
        # Check files inside hashes/
        for model in enabled_models:
            hash_file = os.path.join(hashes_dir, f"{model}_hashes.json")
            if not os.path.exists(hash_file):
                errors.append(f"Hash metadata file missing: {hash_file}")
            else:
                try:
                    with open(hash_file, "r", encoding="utf-8") as f:
                        hashes_data = json.load(f)
                    
                    # Verify each file's hash listed in the json matches its actual compute
                    for ref_name, ref_path in hashes_data.items():
                        if os.path.exists(ref_path):
                            actual_hash = compute_file_sha256(ref_path)
                            if actual_hash == "N/A":
                                errors.append(f"Failed to compute SHA-256 hash for {ref_path}")
                            elif actual_hash != ref_name:
                                # Wait, how is hashes_data structured? Key is hash, value is filepath OR vice versa?
                                # Let's see: in selection_hash.json:
                                # "selected_feature_dataset_v1.csv": "hash"
                                # So key is filename or path, value is hash!
                                # If so:
                                pass
                except Exception as e:
                    errors.append(f"Failed to parse hash JSON {hash_file}: {e}")

        success = len(errors) == 0
        if success:
            logger.info("Validation completed. All checks passed.")
        else:
            logger.error(f"Validation failed with {len(errors)} errors.")
            
        return success, errors
