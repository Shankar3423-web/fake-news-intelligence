import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from ml.training.training_config import TrainingConfig

logger = logging.getLogger("model_training_pipeline")

class MetadataManager:
    """
    Manages metadata generation and storage for trained models.
    Supports reading latest dataset version and feature selection version,
    and saves individual metadata profiles in centralized and localized folders.
    """
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def get_latest_versions(self) -> Tuple[str, str]:
        """
        Parses feature selection versions JSON to retrieve dataset and feature selection versions.
        
        Returns:
            Tuple: (dataset_version: str, feature_selection_version: str)
        """
        selection_versions_path = self.config.get_input_path("selection_versions_json")
        
        dataset_version = "1.0.1" # default fallback
        selection_version = "1" # default fallback
        
        if os.path.exists(selection_versions_path):
            try:
                with open(selection_versions_path, "r", encoding="utf-8") as f:
                    runs = json.load(f)
                if isinstance(runs, list) and len(runs) > 0:
                    last_run = runs[-1]
                    # Check if 'version' exists
                    if "version" in last_run:
                        selection_version = str(last_run["version"])
                    
                    # Try to trace dataset version by checking files hash or references
                    # If selection_versions.json doesn't list dataset_version directly,
                    # check preprocessing versions
                    preprocess_versions_path = "ml/preprocessing/statistics/preprocessing_versions.json"
                    if os.path.exists(preprocess_versions_path):
                        with open(preprocess_versions_path, "r", encoding="utf-8") as pf:
                            pruns = json.load(pf)
                        if isinstance(pruns, list) and len(pruns) > 0:
                            dataset_version = str(pruns[-1].get("Version", "1.0.1"))
            except Exception as e:
                logger.warning(f"Error parsing version files for metadata tracing: {e}. Using defaults.")
                
        return dataset_version, selection_version

    def save_centralized_metadata(self, model_name: str, metadata: Dict[str, Any]) -> str:
        """
        Saves a copy of the model metadata to the centralized metadata directory.
        
        Returns:
            The path to the saved file.
        """
        meta_dir = self.config.get_output_dir("metadata_dir")
        os.makedirs(meta_dir, exist_ok=True)
        
        meta_path = os.path.join(meta_dir, f"{model_name}_metadata.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Saved centralized metadata for '{model_name}' to {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save centralized metadata for {model_name}: {e}")
            raise
            
        return meta_path
