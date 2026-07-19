"""
hash_generator.py
Computes SHA-256 checksums for Phase 12 retraining artifacts.
"""
import hashlib
import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger("retraining_pipeline")


def compute_sha256(file_path: str) -> str:
    """
    Returns the SHA-256 hex-digest of a file, or ``"NOT_FOUND"`` if missing.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    if not os.path.exists(file_path):
        return "NOT_FOUND"
    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(4096), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as exc:  # noqa: BLE001
        logger.error("Hash computation failed for %s: %s", file_path, exc)
        return "ERROR"


class HashGenerator:
    """
    Generates and persists SHA-256 checksums for retraining artifacts.

    Produced artifact: ``retraining_hashes.json``
    """

    def __init__(self, hashes_dir: str) -> None:
        self._hashes_dir = hashes_dir

    def generate(
        self,
        files: Dict[str, str],
        output_filename: str = "retraining_hashes.json",
    ) -> str:
        """
        Computes SHA-256 hashes for all files in *files* and saves results.

        Args:
            files:           Mapping of ``label → file_path``.
            output_filename: JSON filename inside ``hashes_dir``.

        Returns:
            Path to the generated hashes JSON file.
        """
        os.makedirs(self._hashes_dir, exist_ok=True)

        hashes: Dict[str, str] = {}
        for label, path in files.items():
            hashes[label] = compute_sha256(path)
            logger.debug("SHA-256[%s] = %s", label, hashes[label])

        output_path = os.path.join(self._hashes_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump({"files": hashes}, fh, indent=4)

        logger.info("Hashes written to: %s", output_path)
        return output_path
