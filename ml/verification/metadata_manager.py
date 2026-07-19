import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger("verification_pipeline")

class MetadataManager:
    """
    MetadataManager compiles and updates metadata files for the verification runs.
    """
    def __init__(self, metadata_dir: str = "ml/verification/metadata") -> None:
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok=True)

    def save_metadata(
        self,
        file_path: str,
        providers_used: List[str],
        api_response_count: int,
        verification_version: str,
        prediction_version: str,
        model_used: str
    ) -> Dict[str, Any]:
        """
        Saves structured verification run metadata to a JSON file.
        """
        metadata = {
            "providers_used": providers_used,
            "api_response_count": api_response_count,
            "verification_version": verification_version,
            "prediction_version": prediction_version,
            "model_used": model_used,
            "timestamp": datetime.now().isoformat()
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved verification run metadata to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write metadata file at {file_path}: {e}")

        return metadata
