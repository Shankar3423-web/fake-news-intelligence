import logging
import pandas as pd
import numpy as np
import scipy.sparse
from typing import List, Dict, Any
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from ml.feature_selection.selection_utils import BaseFeatureSelector

logger = logging.getLogger("feature_selection_pipeline")

class VarianceSelector(BaseFeatureSelector):
    """
    Selects features based on a variance threshold.
    Min-Max scales the dense features internally to place them on the same [0, 1] range as TF-IDF features.
    """
    def __init__(self, threshold: float = 0.001) -> None:
        self.threshold = threshold
        self.selected_features: List[str] = []
        self.feature_rankings: Dict[str, float] = {}
        self.dense_selector = VarianceThreshold(threshold=threshold)
        self.sparse_selector = VarianceThreshold(threshold=threshold)
        self.scaler = MinMaxScaler()

    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'VarianceSelector':
        logger.info(f"Fitting VarianceSelector with threshold={self.threshold}...")
        
        # 1. Process dense features (scale to [0,1] first so variance is comparable)
        if len(dense_names) > 0:
            X_dense_scaled = self.scaler.fit_transform(X_dense)
            self.dense_selector.fit(X_dense_scaled)
            
            # Extract variances and selected indices
            dense_variances = self.dense_selector.variances_
            dense_support = self.dense_selector.get_support()
            
            for name, var, keep in zip(dense_names, dense_variances, dense_support):
                # Save variance as ranking (rounded to 6 decimals)
                self.feature_rankings[name] = float(var)
                if keep:
                    self.selected_features.append(name)
        
        # 2. Process sparse features (TF-IDF is already in [0,1])
        if len(sparse_names) > 0:
            self.sparse_selector.fit(X_sparse)
            
            # Extract variances and selected indices
            sparse_variances = self.sparse_selector.variances_
            sparse_support = self.sparse_selector.get_support()
            
            for name, var, keep in zip(sparse_names, sparse_variances, sparse_support):
                self.feature_rankings[name] = float(var)
                if keep:
                    self.selected_features.append(name)
                    
        logger.info(f"VarianceSelector completed: selected {len(self.selected_features)} features out of {len(dense_names) + len(sparse_names)}.")
        return self

    def get_selected_features(self) -> List[str]:
        return self.selected_features

    def get_feature_rankings(self) -> Dict[str, float]:
        return self.feature_rankings
