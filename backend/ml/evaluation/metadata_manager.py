import os
import json
import logging
from typing import Dict, Any, Tuple
from ml.evaluation.evaluation_config import EvaluationConfig

logger = logging.getLogger("model_evaluation_pipeline")

class MetadataManager:
    """
    Manages metadata generation and storage for model evaluations.
    Supports reading latest training run details and saving evaluation metadata copies.
    """
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def get_latest_training_info(self) -> Tuple[str, str, str]:
        """
        Parses training versions or registry to trace the latest training version,
        dataset version, and feature selection version.
        
        Returns:
            Tuple of: (training_version: str, dataset_version: str, feature_selection_version: str)
        """
        training_versions_path = self.config.get_input_path("training_versions_json")
        
        training_ver = "training_v1"
        dataset_ver = "1.0.1"
        selection_ver = "1"
        
        if os.path.exists(training_versions_path):
            try:
                with open(training_versions_path, "r", encoding="utf-8") as f:
                    runs = json.load(f)
                if isinstance(runs, list) and len(runs) > 0:
                    last_run = runs[-1]
                    training_ver = last_run.get("training_version", "training_v1")
                    dataset_ver = last_run.get("dataset_version", "1.0.1")
                    selection_ver = last_run.get("feature_selection_version", "1")
            except Exception as e:
                logger.warning(f"Error parsing training versions JSON: {e}. Using defaults.")
                
        return training_ver, dataset_ver, selection_ver

    def save_metadata(self, model_key: str, metadata: Dict[str, Any]) -> str:
        """
        Saves a copy of the model evaluation metadata to the metadata directory.
        
        Returns:
            The path to the saved metadata file.
        """
        meta_dir = self.config.get_output_dir("metadata_dir")
        os.makedirs(meta_dir, exist_ok=True)
        
        meta_path = os.path.join(meta_dir, f"{model_key}_evaluation_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved evaluation metadata for '{model_key}' to {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation metadata for {model_key}: {e}")
            raise
            
        return meta_path
