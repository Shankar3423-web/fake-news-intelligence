import os
import json
import hashlib
import logging
import joblib
import pandas as pd
import numpy as np
import scipy.sparse
from typing import Tuple, List, Dict, Any, Union
from ml.feature_selection.selection_config import SelectionConfig

logger = logging.getLogger("feature_selection_pipeline")

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

def load_data(config: SelectionConfig) -> Tuple[pd.DataFrame, pd.DataFrame, scipy.sparse.csr_matrix, List[str], List[str]]:
    """
    Loads input dense dataset, TF-IDF sparse matrix, and TF-IDF vectorizer.
    Constructs feature names for dense and sparse features.
    
    Returns:
        df_base: DataFrame containing metadata/targets (id, label, cleaned_text)
        df_dense: DataFrame containing dense engineered features
        X_sparse: CSR sparse matrix of TF-IDF features
        dense_names: List of dense feature names
        sparse_names: List of sparse TF-IDF feature names (prefixed with tfidf_)
    """
    input_csv = config.get_path("input_csv")
    input_tfidf_matrix = config.get_path("input_tfidf_matrix")
    input_tfidf_vectorizer = config.get_path("input_tfidf_vectorizer")
    
    logger.info(f"Loading dense dataset from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Extract base columns and dense features
    base_cols = ["id", "label", "cleaned_text"]
    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Required base column '{col}' missing from input dataset.")
            
    df_base = df[base_cols].copy()
    
    # Dense features are all numeric columns excluding label and id
    dense_cols = [col for col in df.columns if col not in base_cols]
    # Check that dense cols are indeed numeric
    df_dense = df[dense_cols].select_dtypes(include=[np.number]).copy()
    dense_names = list(df_dense.columns)
    
    logger.info(f"Loaded {len(df)} samples with {len(dense_names)} dense engineered features.")
    
    logger.info(f"Loading TF-IDF sparse matrix from {input_tfidf_matrix}...")
    X_sparse = joblib.load(input_tfidf_matrix)
    if not scipy.sparse.issparse(X_sparse):
        logger.warning("Loaded TF-IDF matrix is not a sparse matrix. Converting to CSR matrix.")
        X_sparse = scipy.sparse.csr_matrix(X_sparse)
    else:
        # Convert to CSR if not already, to ensure optimal column slicing and math operations
        X_sparse = X_sparse.tocsr()
        
    if X_sparse.shape[0] != len(df):
        raise ValueError(f"TF-IDF matrix rows ({X_sparse.shape[0]}) does not match dense dataset row count ({len(df)}).")
        
    logger.info(f"Loaded TF-IDF matrix of shape {X_sparse.shape}.")
    
    # Load vectorizer to get feature names
    logger.info(f"Loading TF-IDF vectorizer from {input_tfidf_vectorizer} to extract vocabulary...")
    vectorizer = joblib.load(input_tfidf_vectorizer)
    
    try:
        if hasattr(vectorizer, "get_feature_names_out"):
            vocabulary = vectorizer.get_feature_names_out()
        else:
            vocabulary = vectorizer.get_feature_names()
        sparse_names = [f"tfidf_{name}" for name in vocabulary]
    except Exception as e:
        logger.warning(f"Could not extract vocabulary names from vectorizer: {e}. Generating default names.")
        sparse_names = [f"tfidf_feat_{i}" for i in range(X_sparse.shape[1])]
        
    # Ensure sparse_names length matches tfidf matrix shape
    if len(sparse_names) != X_sparse.shape[1]:
        logger.warning(f"Vocabulary size ({len(sparse_names)}) mismatch with sparse matrix columns ({X_sparse.shape[1]}). Truncating/padding names.")
        if len(sparse_names) > X_sparse.shape[1]:
            sparse_names = sparse_names[:X_sparse.shape[1]]
        else:
            sparse_names += [f"tfidf_feat_{i}" for i in range(len(sparse_names), X_sparse.shape[1])]
            
    return df_base, df_dense, X_sparse, dense_names, sparse_names

from abc import ABC, abstractmethod

class BaseFeatureSelector(ABC):
    """
    Abstract Base Class for all feature selection strategies.
    Defines the contract for fitting, selecting, and ranking features.
    """
    @abstractmethod
    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'BaseFeatureSelector':
        """
        Fits the selector on the dense and sparse feature spaces.
        """
        pass

    @abstractmethod
    def get_selected_features(self) -> List[str]:
        """
        Returns the list of selected feature names.
        """
        pass

    @abstractmethod
    def get_feature_rankings(self) -> Dict[str, float]:
        """
        Returns a dictionary mapping feature names to their rankings/importances.
        Higher value indicates more important feature.
        """
        pass

