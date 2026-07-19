import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger("admin_review_pipeline")

class VersionManager:
    """
    VersionManager tracks historical snapshots of review hashes and configurations,
    maintaining version increments.
    """
    def __init__(self, versions_file_path: str) -> None:
        self.versions_file_path = versions_file_path

    def load_versions(self) -> List[Dict[str, Any]]:
        """
        Loads all registered versions from the JSON database.
        """
        if not os.path.exists(self.versions_file_path):
            return []

        try:
            with open(self.versions_file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    return content
                return []
        except Exception as e:
            logger.error(f"Error loading admin review versions registry: {e}")
            return []

    def register_version(self, hashes: Dict[str, str], system_version: str = "1.0.0") -> str:
        """
        Registers a new version snapshot based on the current hashes,
        appends it to the versions list, saves, and returns the version string.
        """
        versions = self.load_versions()
        
        # Calculate next version number
        next_ver_num = len(versions) + 1
        version_str = f"review_v{next_ver_num}"

        entry = {
            "version": version_str,
            "system_version": system_version,
            "timestamp": datetime.now().isoformat(),
            "hashes": hashes
        }

        versions.append(entry)

        os.makedirs(os.path.dirname(self.versions_file_path), exist_ok=True)
        try:
            with open(self.versions_file_path, "w", encoding="utf-8") as f:
                json.dump(versions, f, indent=4)
            logger.info(f"Registered version '{version_str}' in versions registry.")
        except Exception as e:
            logger.error(f"Failed to register version: {e}")
            raise

        return version_str
