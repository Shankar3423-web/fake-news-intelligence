import os
import json
import logging
from typing import Dict
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_utils import compute_file_sha256

logger = logging.getLogger("model_evaluation_pipeline")

class HashGenerator:
    """
    Computes SHA-256 checksums of evaluation reports, statistics, leaderboards,
    predictions, and metadata, storing digests under hashes/evaluation_hashes.json.
    """
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def generate_hashes(self, file_paths: Dict[str, str]) -> str:
        """
        Computes SHA-256 hashes for the generated files.
        Saves the hashes to <hashes_dir>/evaluation_hashes.json.
        
        Returns:
            The path of the generated hashes file.
        """
        hashes_dir = self.config.get_output_dir("hashes_dir")
        os.makedirs(hashes_dir, exist_ok=True)
        
        hashes_data = {}
        for key, path in file_paths.items():
            if os.path.exists(path):
                file_hash = compute_file_sha256(path)
                hashes_data[file_hash] = path  # Key is hash, value is path
                
        hash_file_path = self.config.get_output_path("hashes_file")
        try:
            with open(hash_file_path, "w", encoding="utf-8") as f:
                json.dump(hashes_data, f, indent=4)
            logger.info(f"Generated evaluation hashes at {hash_file_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation hashes: {e}")
            raise
            
        return hash_file_path
