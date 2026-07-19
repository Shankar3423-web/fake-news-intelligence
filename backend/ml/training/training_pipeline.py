import os
import time
import logging
from typing import Dict, Any, List, Tuple
from ml.training.training_config import TrainingConfig
from ml.training.training_logger import setup_logger
from ml.training.training_utils import get_memory_usage, get_library_versions, compute_file_sha256
from ml.training.dataset_loader import DatasetLoader
from ml.training.dataset_splitter import DatasetSplitter
from ml.training.trainer_factory import TrainerFactory
from ml.training.model_registry import ModelRegistry
from ml.training.metadata_manager import MetadataManager
from ml.training.hash_generator import HashGenerator
from ml.training.version_manager import VersionManager
from ml.training.report_generator import ReportGenerator
from ml.training.training_validator import TrainingValidator
from ml.training.training_statistics import TrainingStatistics
from ml.training.training_profiler import TrainingProfiler

class TrainingPipeline:
    """
    Orchestrates the entire Phase 6: Model Training Pipeline.
    Loads selected features, splits dataset, trains classifiers,
    serializes files, indexes in registry, writes metadata,
    generates statistics, compiles report, hashes deliverables,
    logs versions, and validates run integrity.
    """
    def __init__(self, config_path: str = "ml/training/training_config.yaml") -> None:
        self.config = TrainingConfig(config_path)
        self.logger = setup_logger(self.config.get_output_dir("logs_dir"))

    def run(self) -> bool:
        """
        Executes the entire training pipeline.
        
        Returns:
            bool: True if completed successfully without validation failures, False otherwise.
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 6 MODEL TRAINING PIPELINE")
        self.logger.info("=" * 80)
        
        pipeline_start_time = time.perf_counter()
        start_rss, start_peak = get_memory_usage()
        
        warnings: List[str] = []
        model_durations: Dict[str, float] = {}
        model_summaries: List[Dict[str, Any]] = []
        saved_files: Dict[str, str] = {}
        trained_model_ids: List[str] = []
        
        try:
            # 1. Load Dataset
            self.logger.info("--- Step 1: Loading Dataset ---")
            dataset_csv = self.config.get_input_path("dataset_csv")
            feature_names_json = self.config.get_input_path("feature_names_json")
            
            loader = DatasetLoader(dataset_csv, feature_names_json)
            df, selected_features = loader.load_and_validate()
            
            # 2. Split Dataset
            self.logger.info("--- Step 2: Splitting Dataset ---")
            splitter = DatasetSplitter(
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=self.config.stratify,
                shuffle=self.config.shuffle
            )
            X_train, X_test, y_train, y_test = splitter.split(df, selected_features)
            split_info = splitter.get_split_info()
            
            # Validate class imbalance warnings
            train_ratio_fake = split_info["label_distribution"]["train"].get(1, 0) / len(X_train)
            if abs(train_ratio_fake - 0.5) > 0.15:
                warn_msg = f"Training class distribution shows slight imbalance (Fake ratio: {train_ratio_fake:.2%})."
                self.logger.warning(warn_msg)
                warnings.append(warn_msg)

            # 3. Train and Save Enabled Models
            self.logger.info("--- Step 3: Model Training ---")
            models_dir = self.config.get_output_dir("models_dir")
            os.makedirs(models_dir, exist_ok=True)
            
            # Fetch versions from Selection versions
            meta_manager = MetadataManager(self.config)
            dataset_ver, selection_ver = meta_manager.get_latest_versions()
            
            models_to_train = ["logistic_regression", "svm", "random_forest", "xgboost"]
            
            for model_key in models_to_train:
                if not self.config.is_model_enabled(model_key):
                    self.logger.info(f"Model '{model_key}' is disabled in configuration. Skipping.")
                    continue
                    
                self.logger.info(f"Training and saving classifier: '{model_key}'")
                hyperparams = self.config.get_model_hyperparameters(model_key)
                
                # Use Profiler
                with TrainingProfiler.profile(f"{model_key.upper()} Training Phase") as benchmark:
                    trainer = TrainerFactory.create_trainer(model_key, hyperparams)
                    # Train
                    summary = trainer.train(X_train, y_train, split_info)
                    
                    # Update summary with versions
                    summary["dataset_version"] = dataset_ver
                    summary["feature_selection_version"] = selection_ver
                    
                    # Save
                    trainer_saved_files = trainer.save(models_dir)
                    
                    # Map files in saved files
                    for k, path in trainer_saved_files.items():
                        saved_files[f"{model_key}_{k}"] = path
                        
                model_durations[model_key] = benchmark.get("duration_sec", 0.0)
                model_summaries.append(summary)
                
                # Register in trained IDs list
                model_run_id = f"model_{model_key}_{datetime_suffix()}"
                trained_model_ids.append(model_run_id)

            # 4. Generate Model Registry
            self.logger.info("--- Step 4: Updating Model Registry ---")
            registry_path = self.config.get_output_path("registry_file")
            registry_manager = ModelRegistry(registry_path)
            
            # Training run version matches next version in registry or version manager
            # Let's read current version list to determine version code
            version_manager = VersionManager(self.config)
            runs_history = version_manager.load_versions()
            next_training_ver_num = len(runs_history) + 1
            training_ver_str = f"training_v{next_training_ver_num}"
            
            for summary, model_id in zip(model_summaries, trained_model_ids):
                model_folder_rel = os.path.join(models_dir, summary["algorithm"].lower().replace(" ", "_"))
                registry_manager.register_model(
                    model_id=model_id,
                    algorithm=summary["algorithm"],
                    version=training_ver_str,
                    dataset_version=dataset_ver,
                    training_date=split_info["timestamp"],
                    feature_count=summary["feature_count"],
                    training_samples=summary["samples_trained"],
                    testing_samples=split_info["testing_samples"],
                    path=model_folder_rel
                )
            saved_files["registry_file"] = registry_path

            # 5. Centralized Metadata Storage
            self.logger.info("--- Step 5: Generating Metadata Copies ---")
            for summary in model_summaries:
                # Compile localized metadata dict
                lib_versions = get_library_versions()
                test_size = split_info.get("config", {}).get("test_size", 0.2)
                train_test_ratio = f"{int((1 - test_size) * 100)}/{int(test_size * 100)}"
                
                meta_dict = {
                    "model_name": summary["algorithm"].lower().replace(" ", "_"),
                    "algorithm": summary["algorithm"],
                    "training_date": split_info["timestamp"],
                    "training_time": summary["training_duration_sec"],
                    "memory_used_mb": summary["memory_used_mb"],
                    "feature_count": summary["feature_count"],
                    "dataset_version": dataset_ver,
                    "feature_selection_version": selection_ver,
                    "random_seed": summary["split_info"]["config"]["random_state"],
                    "hyperparameters": summary["split_info"]["hyperparameters"] if "hyperparameters" in summary["split_info"] else summary.get("hyperparameters", {}),
                    "train_test_ratio": train_test_ratio,
                    "training_samples": summary["samples_trained"],
                    "testing_samples": split_info["testing_samples"],
                    "python_version": lib_versions.get("python", ""),
                    "library_versions": lib_versions
                }
                meta_copy_path = meta_manager.save_centralized_metadata(meta_dict["model_name"], meta_dict)
                saved_files[f"{meta_dict['model_name']}_central_metadata"] = meta_copy_path

            # 6. Generate Training Statistics
            self.logger.info("--- Step 6: Compiling Statistics ---")
            pipeline_end_time = time.perf_counter()
            total_duration = pipeline_end_time - pipeline_start_time
            end_rss, end_peak = get_memory_usage()
            peak_memory_used = max(0.0, end_peak - start_peak)
            
            stats_path = self.config.get_output_path("statistics_file")
            stats_manager = TrainingStatistics()
            stats_manager.generate(
                dataset_path=dataset_csv,
                feature_count=len(selected_features),
                training_rows=len(X_train),
                testing_rows=len(X_test),
                total_pipeline_duration=total_duration,
                peak_memory_used_mb=peak_memory_used,
                model_durations=model_durations
            )
            stats_manager.save(stats_path)
            saved_files["statistics_file"] = stats_path

            # 7. Generate Hashing for Models
            self.logger.info("--- Step 7: Generating SHA-256 Hashes ---")
            hash_gen = HashGenerator(self.config)
            
            # Generate hashes for each model folder files
            for model_key in models_to_train:
                if not self.config.is_model_enabled(model_key):
                    continue
                model_files_map = {
                    "model.joblib": os.path.join(models_dir, model_key, "model.joblib"),
                    "metadata.json": os.path.join(models_dir, model_key, "metadata.json"),
                    "training_config.json": os.path.join(models_dir, model_key, "training_config.json"),
                    "feature_order.json": os.path.join(models_dir, model_key, "feature_order.json")
                }
                # Filter present
                model_files_map = {k: v for k, v in model_files_map.items() if os.path.exists(v)}
                hash_path = hash_gen.generate_model_hashes(model_key, model_files_map)
                saved_files[f"{model_key}_hashes"] = hash_path

            # 8. Generate Reports
            self.logger.info("--- Step 8: Generating Training Report ---")
            report_path = self.config.get_output_path("report_file")
            report_gen = ReportGenerator(report_path)
            
            dataset_info = {
                "path": dataset_csv,
                "size_bytes": os.path.getsize(dataset_csv),
                "feature_count": len(selected_features)
            }
            report_gen.generate(
                dataset_info=dataset_info,
                split_info=split_info,
                model_summaries=model_summaries,
                pipeline_duration=total_duration,
                generated_files=saved_files,
                warnings=warnings
            )
            saved_files["report_file"] = report_path

            # Hashes for central pipeline deliverables
            pipeline_files_to_hash = {
                "registry.json": registry_path,
                "training_statistics.json": stats_path,
                "training_report.md": report_path
            }
            pipeline_hash_file = hash_gen.generate_pipeline_hashes(pipeline_files_to_hash)
            saved_files["pipeline_hashes"] = pipeline_hash_file
            
            pipeline_digest = compute_file_sha256(pipeline_hash_file)

            # 9. Log Training Version
            self.logger.info("--- Step 9: Registering Training Run Version ---")
            version_entry = version_manager.register_run(
                dataset_version=dataset_ver,
                feature_selection_version=selection_ver,
                model_ids=trained_model_ids,
                pipeline_hash=pipeline_digest,
                files_dict=saved_files
            )
            versions_path = self.config.get_output_path("versions_file")
            saved_files["versions_file"] = versions_path

            # 10. Pipeline Validation
            self.logger.info("--- Step 10: Running Integrity Validation ---")
            validator = TrainingValidator(self.config)
            valid, validation_errors = validator.validate_all()
            
            if not valid:
                self.logger.error("Pipeline validation encountered errors:")
                for err in validation_errors:
                    self.logger.error(f"- {err}")
                return False

            self.logger.info("=" * 80)
            self.logger.info("PHASE 6 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            return True

        except Exception as e:
            self.logger.exception(f"Unhandled exception in Phase 6 Model Training Pipeline: {e}")
            raise

def datetime_suffix() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
