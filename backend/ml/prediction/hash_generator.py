import os
import json
import hashlib
import logging
from typing import Dict

logger = logging.getLogger("prediction_pipeline")

class HashGenerator:
    """
    Generates and maintains SHA-256 hashes for all prediction engine output files.
    """
    def __init__(self, hash_file_path: str = "ml/prediction/hashes/prediction_hashes.json") -> None:
        self.hash_file_path = hash_file_path
        os.makedirs(os.path.dirname(self.hash_file_path), exist_ok=True)

    def _compute_sha256(self, file_path: str) -> str:
        """Computes the SHA-256 hash of a file."""
        if not os.path.exists(file_path):
            return "N/A"
            
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return "N/A"

    def generate_hashes(self, files_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Computes SHA-256 hashes for the provided dictionary of file labels to paths.
        Saves the hashes to prediction_hashes.json.
        """
        logger.info("Generating SHA-256 signatures for prediction outputs...")
        hashes = {}
        for key, path in files_dict.items():
            if path and os.path.exists(path):
                hashes[key] = self._compute_sha256(path)
                
        try:
            with open(self.hash_file_path, "w", encoding="utf-8") as f:
                json.dump(hashes, f, indent=4)
            logger.info(f"Hashes saved successfully to {self.hash_file_path}")
        except Exception as e:
            logger.error(f"Failed to save hashes to {self.hash_file_path}: {e}")
            raise e
            
        return hashes
