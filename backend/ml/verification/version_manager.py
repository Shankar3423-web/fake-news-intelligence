import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger("verification_pipeline")

class VersionManager:
    """
    VersionManager tracks historical runs and assigns sequential version strings
    (e.g., verification_v1, verification_v2) associated with each pipeline run.
    """
    def __init__(self, versions_file_path: str = "ml/verification/versions/verification_versions.json") -> None:
        self.versions_file_path = versions_file_path
        os.makedirs(os.path.dirname(self.versions_file_path), exist_ok=True)

    def load_versions(self) -> List[Dict[str, Any]]:
        """
        Loads the versions history list from the JSON file.
        """
        if not os.path.exists(self.versions_file_path):
            return []
        try:
            with open(self.versions_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Could not load versions from {self.versions_file_path}: {e}. Returning empty.")
            return []

    def register_version(
        self,
        prediction_version: str,
        model_used: str,
        hashes: Dict[str, str]
    ) -> str:
        """
        Registers a new version entry and writes the updated list back to disk.
        Returns the version string.
        """
        versions = self.load_versions()
        version_num = len(versions) + 1
        version_str = f"verification_v{version_num}"

        entry = {
            "version": version_str,
            "prediction_version": prediction_version,
            "model_used": model_used,
            "hashes": hashes,
            "timestamp": datetime.now().isoformat()
        }
        
        versions.append(entry)

        try:
            with open(self.versions_file_path, "w", encoding="utf-8") as f:
                json.dump(versions, f, indent=2, ensure_ascii=False)
            logger.info(f"Registered verification version '{version_str}' in {self.versions_file_path}")
        except Exception as e:
            logger.error(f"Failed to write version database at {self.versions_file_path}: {e}")

        return version_str
