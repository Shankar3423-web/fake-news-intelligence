import os
import json
import hashlib
import logging
from typing import Dict

logger = logging.getLogger("verification_pipeline")

class HashGenerator:
    """
    HashGenerator computes SHA-256 integrity checksums for artifacts, reports,
    and statistics, saving them in a central JSON registry.
    """
    def __init__(self, hash_file_path: str = "ml/verification/hashes/verification_hashes.json") -> None:
        self.hash_file_path = hash_file_path
        os.makedirs(os.path.dirname(self.hash_file_path), exist_ok=True)

    def generate_hashes(self, file_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Computes SHA-256 hashes for all existing files in the file_paths mapping
        and saves the collection registry.
        """
        hashes = {}
        for key, path in file_paths.items():
            if not os.path.exists(path):
                logger.warning(f"File {path} does not exist. Skipping hash computation.")
                continue
                
            try:
                hasher = hashlib.sha256()
                with open(path, "rb") as f:
                    buf = f.read()
                    hasher.update(buf)
                hashes[key] = hasher.hexdigest()
            except Exception as e:
                logger.error(f"Error generating hash for file '{path}': {e}")

        # Save registry
        try:
            with open(self.hash_file_path, "w", encoding="utf-8") as f:
                json.dump(hashes, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved SHA-256 hashes registry at {self.hash_file_path}")
        except Exception as e:
            logger.error(f"Failed to write hashes registry file: {e}")

        return hashes
