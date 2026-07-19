import os
import sys
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
import pandas as pd
from ml.evaluation.evaluation_config import EvaluationConfig

logger = logging.getLogger("model_evaluation_pipeline")

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
    
    libs = ["scikit-learn", "xgboost", "joblib", "pandas", "numpy", "yaml", "matplotlib"]
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

class BaseModelEvaluator(ABC):
    """
    Abstract base class for all model evaluators in Phase 7.
    Defines the contract for model loading, evaluation, and sizing.
    """
    def __init__(self, model_key: str, algorithm_name: str, config: EvaluationConfig) -> None:
        self.model_key = model_key
        self.algorithm_name = algorithm_name
        self.config = config
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}
        self.feature_order: List[str] = []

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Loads the model, generates predictions, calculates metrics, and profiles performance.
        Returns a dictionary of results.
        """
        pass

