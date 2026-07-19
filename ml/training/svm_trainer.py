import os
import json
import time
import logging
import joblib
import pandas as pd
from typing import Dict, Any, List
from ml.training.training_utils import BaseModelTrainer, get_memory_usage, get_library_versions

logger = logging.getLogger("model_training_pipeline")

class SvmTrainer(BaseModelTrainer):
    """
    Trainer for Linear SVM (LinearSVC) classifier.
    Handles hyperparameter initialization, fitting, performance measuring,
    and writing serialized models/metadata.
    """
    def __init__(self, hyperparameters: Dict[str, Any]) -> None:
        super().__init__(
            model_name="svm",
            algorithm_name="Linear SVM",
            hyperparameters=hyperparameters
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trains a Linear SVM model.
        """
        from sklearn.svm import LinearSVC
        
        logger.info("Training Linear SVM model...")
        self.feature_order = list(X_train.columns)
        
        start_time = time.perf_counter()
        mem_before_rss, mem_before_peak = get_memory_usage()
        
        # Instantiate and fit
        self.model = LinearSVC(**self.hyperparameters)
        self.model.fit(X_train, y_train)
        
        end_time = time.perf_counter()
        mem_after_rss, mem_after_peak = get_memory_usage()
        
        duration = round(end_time - start_time, 4)
        mem_used = round(max(0.0, mem_after_rss - mem_before_rss), 2)
        
        self.training_summary = {
            "algorithm": self.algorithm_name,
            "training_duration_sec": duration,
            "memory_rss_before_mb": mem_before_rss,
            "memory_rss_after_mb": mem_after_rss,
            "memory_used_mb": mem_used,
            "feature_count": len(self.feature_order),
            "samples_trained": len(X_train),
            "split_info": split_info
        }
        
        logger.info(f"Linear SVM training complete in {duration} seconds. Memory used: {mem_used} MB.")
        return self.training_summary

    def save(self, output_dir: str) -> Dict[str, str]:
        """
        Saves the model and its metadata into outputs/models/svm/.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        model_dir = os.path.join(output_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "model.joblib")
        config_path = os.path.join(model_dir, "training_config.json")
        features_path = os.path.join(model_dir, "feature_order.json")
        metadata_path = os.path.join(model_dir, "metadata.json")
        
        logger.info(f"Saving Linear SVM files to {model_dir}...")
        
        # Save model object
        joblib.dump(self.model, model_path)
        
        # Save exact hyperparameter config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.hyperparameters, f, indent=4)
            
        # Save feature order
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(self.feature_order, f, indent=4)
            
        # Compile and save metadata
        split_details = self.training_summary.get("split_info", {})
        lib_versions = get_library_versions()
        
        # Calculate train/test ratio
        test_size = split_details.get("config", {}).get("test_size", 0.2)
        train_test_ratio = f"{int((1 - test_size) * 100)}/{int(test_size * 100)}"
        
        metadata = {
            "model_name": self.model_name,
            "algorithm": self.algorithm_name,
            "training_date": split_details.get("timestamp", datetime_placeholder()),
            "training_time": self.training_summary.get("training_duration_sec", 0.0),
            "memory_used_mb": self.training_summary.get("memory_used_mb", 0.0),
            "feature_count": len(self.feature_order),
            "dataset_version": self.training_summary.get("dataset_version", "unknown"),
            "feature_selection_version": self.training_summary.get("feature_selection_version", "unknown"),
            "random_seed": self.hyperparameters.get("random_state", 42),
            "hyperparameters": self.hyperparameters,
            "train_test_ratio": train_test_ratio,
            "training_samples": self.training_summary.get("samples_trained", 0),
            "testing_samples": split_details.get("testing_samples", 0),
            "python_version": lib_versions.get("python", ""),
            "library_versions": lib_versions
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        return {
            "model": model_path,
            "config": config_path,
            "features": features_path,
            "metadata": metadata_path
        }

def datetime_placeholder() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
