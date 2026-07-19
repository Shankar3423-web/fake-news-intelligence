import logging
import pandas as pd
import numpy as np
import scipy.sparse
from typing import List, Dict, Any
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from ml.feature_selection.selection_utils import BaseFeatureSelector

logger = logging.getLogger("feature_selection_pipeline")

class ChiSquareSelector(BaseFeatureSelector):
    """
    Selects top features using the Chi-Square test.
    Applies Min-Max scaling to dense features internally to guarantee non-negativity.
    Separately ranks dense and sparse features to select top_k_dense and top_k_sparse.
    """
    def __init__(self, top_k_dense: int = 15, top_k_sparse: int = 200) -> None:
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.selected_features: List[str] = []
        self.feature_rankings: Dict[str, float] = {}
        self.scaler = MinMaxScaler()

    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'ChiSquareSelector':
        logger.info(f"Fitting ChiSquareSelector (top_k_dense={self.top_k_dense}, top_k_sparse={self.top_k_sparse})...")
        
        # 1. Compute Chi-Square for dense features
        if len(dense_names) > 0:
            logger.info("Computing Chi-Square for dense features...")
            # MinMax scale to guarantee non-negativity
            X_dense_scaled = self.scaler.fit_transform(X_dense)
            
            # Compute chi-square statistic
            chi2_scores, p_values = chi2(X_dense_scaled, y)
            
            dense_rankings = {}
            for name, score in zip(dense_names, chi2_scores):
                # Replace NaN values with 0.0
                score_val = float(score) if not np.isnan(score) else 0.0
                self.feature_rankings[name] = score_val
                dense_rankings[name] = score_val
                
            # Select top K dense features
            sorted_dense = sorted(dense_rankings.keys(), key=lambda k: dense_rankings[k], reverse=True)
            k_dense = min(self.top_k_dense, len(sorted_dense))
            selected_dense = sorted_dense[:k_dense]
            self.selected_features.extend(selected_dense)
            logger.info(f"Selected top {k_dense} dense features using Chi-Square.")

        # 2. Compute Chi-Square for sparse features
        if len(sparse_names) > 0:
            logger.info("Computing Chi-Square for sparse features...")
            # Sparse TF-IDF is already non-negative and scaled
            chi2_scores, p_values = chi2(X_sparse, y)
            
            sparse_rankings = {}
            for name, score in zip(sparse_names, chi2_scores):
                score_val = float(score) if not np.isnan(score) else 0.0
                self.feature_rankings[name] = score_val
                sparse_rankings[name] = score_val
                
            # Select top K sparse features
            sorted_sparse = sorted(sparse_rankings.keys(), key=lambda k: sparse_rankings[k], reverse=True)
            k_sparse = min(self.top_k_sparse, len(sorted_sparse))
            selected_sparse = sorted_sparse[:k_sparse]
            self.selected_features.extend(selected_sparse)
            logger.info(f"Selected top {k_sparse} sparse features using Chi-Square.")

        logger.info(f"ChiSquareSelector completed: selected {len(self.selected_features)} features.")
        return self

    def get_selected_features(self) -> List[str]:
        return self.selected_features

    def get_feature_rankings(self) -> Dict[str, float]:
        return self.feature_rankings
