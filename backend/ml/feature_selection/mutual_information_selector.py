import logging
import pandas as pd
import numpy as np
import scipy.sparse
from typing import List, Dict, Any
from sklearn.feature_selection import mutual_info_classif
from ml.feature_selection.selection_utils import BaseFeatureSelector

logger = logging.getLogger("feature_selection_pipeline")

class MutualInformationSelector(BaseFeatureSelector):
    """
    Selects top features using Mutual Information (MI).
    Separately ranks dense and sparse features to select top_k_dense and top_k_sparse.
    Supports dataset sub-sampling for efficiency.
    """
    def __init__(
        self, 
        top_k_dense: int = 15, 
        top_k_sparse: int = 150, 
        sub_sample_size: int = 10000, 
        random_state: int = 42
    ) -> None:
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.sub_sample_size = sub_sample_size
        self.random_state = random_state
        self.selected_features: List[str] = []
        self.feature_rankings: Dict[str, float] = {}

    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'MutualInformationSelector':
        logger.info(f"Fitting MutualInformationSelector (top_k_dense={self.top_k_dense}, top_k_sparse={self.top_k_sparse})...")
        
        n_samples = len(y)
        
        # Sub-sample data if it exceeds sub_sample_size
        if n_samples > self.sub_sample_size:
            logger.info(f"Sub-sampling dataset from {n_samples} to {self.sub_sample_size} for Mutual Information calculation.")
            np.random.seed(self.random_state)
            indices = np.random.choice(n_samples, self.sub_sample_size, replace=False)
            
            # Sub-sample dense, sparse, and labels
            X_dense_sub = X_dense.iloc[indices]
            X_sparse_sub = X_sparse[indices]
            y_sub = y[indices]
        else:
            X_dense_sub = X_dense
            X_sparse_sub = X_sparse
            y_sub = y

        # 1. Compute Mutual Information for dense features
        if len(dense_names) > 0:
            logger.info("Computing Mutual Information for dense features...")
            dense_mi = mutual_info_classif(
                X_dense_sub, 
                y_sub, 
                random_state=self.random_state
            )
            
            dense_rankings = {}
            for name, score in zip(dense_names, dense_mi):
                self.feature_rankings[name] = float(score)
                dense_rankings[name] = float(score)
                
            # Select top K dense features
            sorted_dense = sorted(dense_rankings.keys(), key=lambda k: dense_rankings[k], reverse=True)
            k_dense = min(self.top_k_dense, len(sorted_dense))
            selected_dense = sorted_dense[:k_dense]
            self.selected_features.extend(selected_dense)
            logger.info(f"Selected top {k_dense} dense features using Mutual Information.")

        # 2. Compute Mutual Information for sparse features
        if len(sparse_names) > 0:
            logger.info("Computing Mutual Information for sparse features...")
            sparse_mi = mutual_info_classif(
                X_sparse_sub, 
                y_sub, 
                random_state=self.random_state
            )
            
            sparse_rankings = {}
            for name, score in zip(sparse_names, sparse_mi):
                self.feature_rankings[name] = float(score)
                sparse_rankings[name] = float(score)
                
            # Select top K sparse features
            sorted_sparse = sorted(sparse_rankings.keys(), key=lambda k: sparse_rankings[k], reverse=True)
            k_sparse = min(self.top_k_sparse, len(sorted_sparse))
            selected_sparse = sorted_sparse[:k_sparse]
            self.selected_features.extend(selected_sparse)
            logger.info(f"Selected top {k_sparse} sparse features using Mutual Information.")
            
        logger.info(f"MutualInformationSelector completed: selected {len(self.selected_features)} features.")
        return self

    def get_selected_features(self) -> List[str]:
        return self.selected_features

    def get_feature_rankings(self) -> Dict[str, float]:
        return self.feature_rankings
