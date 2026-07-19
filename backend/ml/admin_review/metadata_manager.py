import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("admin_review_pipeline")

class MetadataManager:
    """
    MetadataManager creates and updates metadata information for the admin review system.
    """
    def __init__(self, metadata_dir: str) -> None:
        self.metadata_dir = metadata_dir

    def save_metadata(
        self,
        file_path: str,
        pipeline_version: str = "1.0.0",
        system_version: str = "1.0.0",
        configuration_version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Generates and saves the metadata json file.
        """
        metadata = {
            "pipeline_name": "Phase 11 - Admin Review System",
            "pipeline_version": pipeline_version,
            "system_version": system_version,
            "configuration_version": configuration_version,
            "last_executed": datetime.now().isoformat(),
            "status": "OPERATIONAL",
            "environment": {
                "os": os.name,
                "project_root": os.getcwd()
            }
        }

        os.makedirs(self.metadata_dir, exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Admin review metadata saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {file_path}: {e}")
            raise

        return metadata
