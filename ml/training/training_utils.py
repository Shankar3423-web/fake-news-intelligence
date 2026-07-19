import os
import sys
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import pandas as pd

logger = logging.getLogger("model_training_pipeline")

def compute_file_sha256(file_path: str) -> str:
    """Computes the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        if not os.path.exists(file_path):
            return "N/A"
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return "N/A"

def get_memory_usage() -> Tuple[float, float]:
    """
    Returns (current_memory_mb, peak_memory_mb) of the current process.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss = round(mem_info.rss / (1024 * 1024), 2)
        
        # Track peak memory using peak_wset on Windows
        if hasattr(mem_info, 'peak_wset'):
            peak = round(mem_info.peak_wset / (1024 * 1024), 2)
        else:
            peak = rss
        return rss, peak
    except Exception:
        return 0.0, 0.0

def get_library_versions() -> Dict[str, str]:
    """
    Collects versions of Python and key machine learning libraries.
    """
    versions = {
        "python": sys.version.split()[0]
    }
    
    libs = ["scikit-learn", "xgboost", "joblib", "pandas", "numpy", "yaml"]
    for lib in libs:
        try:
            if lib == "scikit-learn":
                import sklearn
                versions[lib] = sklearn.__version__
            elif lib == "yaml":
                import yaml
                versions[lib] = yaml.__version__
            else:
                imported = __import__(lib.replace("-", "_"))
                versions[lib] = imported.__version__
        except ImportError:
            versions[lib] = "not_installed"
            
    return versions

class BaseModelTrainer(ABC):
    """
    Abstract base class defining the contract for all model trainers in Phase 6.
    """
    def __init__(self, model_name: str, algorithm_name: str, hyperparameters: Dict[str, Any]) -> None:
        self.model_name = model_name
        self.algorithm_name = algorithm_name
        self.hyperparameters = hyperparameters
        self.model: Any = None
        self.feature_order: List[str] = []
        self.training_summary: Dict[str, Any] = {}

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, split_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trains the model, measures execution time and memory usage.
        """
        pass

    @abstractmethod
    def save(self, output_dir: str) -> Dict[str, str]:
        """
        Saves the trained model and associated metadata files.
        Returns a dictionary of generated files and their absolute paths.
        """
        pass
