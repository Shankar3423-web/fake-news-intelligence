import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
from ml.evaluation.evaluation_config import EvaluationConfig

logger = logging.getLogger("model_evaluation_pipeline")

class VersionManager:
    """
    Manages the evaluation run version history file (evaluation_versions.json).
    Automatically increments version numbers and registers run parameters,
    hashes, best model ID, and dataset contexts.
    """
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.versions_path = config.get_output_path("versions_file")

    def load_versions(self) -> List[Dict[str, Any]]:
        """
        Loads the list of prior evaluation runs. If the file doesn't exist, returns empty list.
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
        training_version: str,
        dataset_version: str,
        feature_selection_version: str,
        best_model_id: str,
        pipeline_hash: str,
        files_dict: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Appends an evaluation run entry to evaluation_versions.json and saves the file.
        
        Returns:
            The newly created version log entry dictionary.
        """
        runs = self.load_versions()
        
        # Calculate next version number
        next_version_num = 1
        if runs:
            version_nums = [r.get("version", 0) for r in runs if isinstance(r.get("version"), int)]
            if version_nums:
                next_version_num = max(version_nums) + 1
                
        timestamp = datetime.now(timezone.utc).isoformat()
        evaluation_version_str = f"evaluation_v{next_version_num}"
        
        # Build file-to-hash mappings for versioning reference
        from ml.evaluation.evaluation_utils import compute_file_sha256
        file_hashes = {}
        for file_name, file_path in files_dict.items():
            if os.path.exists(file_path):
                file_hashes[file_name] = compute_file_sha256(file_path)

        entry = {
            "version": next_version_num,
            "timestamp": timestamp,
            "evaluation_version": evaluation_version_str,
            "training_version": training_version,
            "dataset_version": dataset_version,
            "feature_selection_version": feature_selection_version,
            "best_model_id": best_model_id,
            "hash": pipeline_hash,
            "files": file_hashes
        }
        
        runs.append(entry)
        self.save_versions(runs)
        
        logger.info(f"Logged evaluation execution version: {next_version_num} (evaluation_version={evaluation_version_str})")
        return entry

    def save_versions(self, versions_list: List[Dict[str, Any]]) -> None:
        """
        Saves the evaluation run version log back to the file.
        """
        parent_dir = os.path.dirname(self.versions_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            with open(self.versions_path, "w", encoding="utf-8") as f:
                json.dump(versions_list, f, indent=4)
            logger.info(f"Saved evaluation version records to {self.versions_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation version records: {e}")
            raise
