import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger("prediction_pipeline")

class VersionManager:
    """
    Manages historical prediction versions and registries inside prediction_versions.json.
    """
    def __init__(self, versions_file_path: str = "ml/prediction/versions/prediction_versions.json") -> None:
        self.versions_file_path = versions_file_path
        os.makedirs(os.path.dirname(self.versions_file_path), exist_ok=True)

    def load_versions(self) -> List[Dict[str, Any]]:
        """Loads version records from prediction_versions.json."""
        if not os.path.exists(self.versions_file_path):
            return []
            
        try:
            with open(self.versions_file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    return content
                return []
        except Exception as e:
            logger.warning(f"Failed to parse version database: {e}. Starting with empty history.")
            return []

    def register_version(
        self,
        training_version: str,
        evaluation_version: str,
        model_used: str,
        hashes: Dict[str, str]
    ) -> str:
        """
        Registers a new execution run in the versions database.
        Increments the Prediction Version and appends metadata + hashes.
        
        Returns:
            The newly created prediction version string.
        """
        logger.info("Registering prediction version in database...")
        history = self.load_versions()
        
        next_ver_num = len(history) + 1
        prediction_version = f"prediction_v{next_ver_num}"
        
        record = {
            "prediction_version": prediction_version,
            "training_version": training_version,
            "evaluation_version": evaluation_version,
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used,
            "hashes": hashes
        }
        
        history.append(record)
        
        try:
            with open(self.versions_file_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            logger.info(f"Registered version: {prediction_version}")
        except Exception as e:
            logger.error(f"Failed to write prediction version: {e}")
            raise e
            
        return prediction_version
