import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from ml.training.training_config import TrainingConfig

logger = logging.getLogger("model_training_pipeline")

class VersionManager:
    """
    Manages the training run version history file (training_versions.json).
    Automatically increments version numbers and registers run parameters,
    hashes, model IDs, and dataset contexts.
    """
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.versions_path = config.get_output_path("versions_file")

    def load_versions(self) -> List[Dict[str, Any]]:
        """
        Loads the list of prior training runs. If the file doesn't exist, returns empty list.
        """
        if not os.path.exists(self.versions_path):
            logger.debug(f"Versions file not found at {self.versions_path}. Creating new log.")
            return []
            
        try:
            with open(self.versions_path, "r", encoding="utf-8") as f:
                versions = json.load(f)
                if not isinstance(versions, list):
                    logger.warning("Versions JSON is not a list. Re-initializing.")
                    return []
                return versions
        except Exception as e:
            logger.error(f"Error loading versions from {self.versions_path}: {e}. Returning empty list.")
            return []

    def register_run(
        self,
        dataset_version: str,
        feature_selection_version: str,
        model_ids: List[str],
        pipeline_hash: str,
        files_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Appends a run entry to training_versions.json and saves the file.
        
        Returns:
            The newly created version log entry dictionary.
        """
        runs = self.load_versions()
        
        # Calculate next version number
        next_version_num = 1
        if runs:
            # Find max version number
            version_nums = [r.get("version", 0) for r in runs if isinstance(r.get("version"), int)]
            if version_nums:
                next_version_num = max(version_nums) + 1
                
        timestamp = datetime.now(timezone.utc).isoformat()
        training_version_str = f"run_{next_version_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Build file-to-hash mappings for versioning reference
        from ml.training.training_utils import compute_file_sha256
        file_hashes = {}
        for file_name, file_path in files_dict.items():
            if os.path.exists(file_path):
                file_hashes[file_name] = compute_file_sha256(file_path)

        entry = {
            "version": next_version_num,
            "timestamp": timestamp,
            "dataset_version": dataset_version,
            "feature_selection_version": feature_selection_version,
            "training_version": training_version_str,
            "hash": pipeline_hash,
            "model_ids": model_ids,
            "files": file_hashes
        }
        
        runs.append(entry)
        self.save_versions(runs)
        
        logger.info(f"Logged training execution version: {next_version_num} (training_version={training_version_str})")
        return entry

    def save_versions(self, versions_list: List[Dict[str, Any]]) -> None:
        """
        Saves the training run version log back to the file.
        """
        parent_dir = os.path.dirname(self.versions_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            with open(self.versions_path, "w", encoding="utf-8") as f:
                json.dump(versions_list, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write versions log to {self.versions_path}: {e}")
            raise
