import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("feedback_pipeline")

class MetadataManager:
    """
    MetadataManager compiles and updates metadata files for the feedback runs.
    """
    def __init__(self, metadata_dir: str = "ml/feedback/metadata") -> None:
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)

    def save_metadata(
        self,
        file_path: str,
        pipeline_version: str,
        feedback_version: str,
        configuration_version: str
    ) -> Dict[str, Any]:
        """
        Saves structured feedback run metadata to a JSON file.
        """
        metadata = {
            "pipeline_version": pipeline_version,
            "feedback_version": feedback_version,
            "timestamp": datetime.now().isoformat(),
            "configuration_version": configuration_version
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved feedback run metadata to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write metadata file at {file_path}: {e}")

        return metadata
