import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("prediction_pipeline")

class MetadataManager:
    """
    Manages the creation and saving of prediction metadata.
    """
    def __init__(self, metadata_dir: str = "ml/prediction/metadata") -> None:
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)

    def save_metadata(
        self,
        file_path: str,
        model_name: str,
        training_version: str,
        evaluation_version: str,
        feature_version: str,
        prediction_version: str,
        pipeline_version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Creates and saves prediction metadata JSON file.
        """
        logger.info(f"Generating prediction metadata at {file_path}...")
        
        metadata = {
            "model_used": model_name,
            "training_version": training_version,
            "evaluation_version": evaluation_version,
            "feature_version": feature_version,
            "prediction_version": prediction_version,
            "pipeline_version": pipeline_version,
            "prediction_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save metadata to {file_path}: {e}")
            raise e
            
        return metadata
