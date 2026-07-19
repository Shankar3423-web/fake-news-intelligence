import os
import json
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger("admin_review_pipeline")

class HashGenerator:
    """
    HashGenerator computes SHA-256 hashes of generated files
    and maintains an integrity registry.
    """
    def __init__(self, hash_file_path: str) -> None:
        self.hash_file_path = hash_file_path

    def calculate_file_hash(self, file_path: str) -> str:
        """
        Calculates the SHA-256 hash of a file.
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found for hashing: {file_path}")
            return ""

        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def generate_hashes(self, files_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Generates SHA-256 hashes for a dictionary of file paths,
        saves the registry to admin_review_hashes.json, and returns the registry.
        """
        hashes = {}
        for key, filepath in files_dict.items():
            if os.path.exists(filepath):
                hashes[key] = self.calculate_file_hash(filepath)
            else:
                hashes[key] = ""

        os.makedirs(os.path.dirname(self.hash_file_path), exist_ok=True)
        try:
            with open(self.hash_file_path, "w", encoding="utf-8") as f:
                json.dump(hashes, f, indent=4)
            logger.info(f"SHA-256 hashes registered and saved to {self.hash_file_path}")
        except Exception as e:
            logger.error(f"Failed to save hashes registry to {self.hash_file_path}: {e}")
            raise

        return hashes
