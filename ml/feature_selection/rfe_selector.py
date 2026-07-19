import logging
import pandas as pd
import numpy as np
import scipy.sparse
from typing import List, Dict, Any
from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from ml.feature_selection.selection_utils import BaseFeatureSelector

logger = logging.getLogger("feature_selection_pipeline")

class RFESelector(BaseFeatureSelector):
    """
    Selects features using Recursive Feature Elimination (RFE).
    Uses Logistic Regression as the estimator.
    To prevent extreme runtimes, a pre-filtering step (using Chi-Square)
    reduces the feature space to pre_selected_pool_size before running RFE.
    """
    def __init__(
        self, 
        n_features_to_select: int = 30,
        step: int = 5,
        random_state: int = 42,
        pre_selected_pool_size: int = 80
    ) -> None:
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.random_state = random_state
        self.pre_selected_pool_size = pre_selected_pool_size
        self.selected_features: List[str] = []
        self.feature_rankings: Dict[str, float] = {}
        self.scaler = MinMaxScaler()
        
        # Fast estimator for RFE
        self.estimator = LogisticRegression(
            max_iter=1000, 
            solver="liblinear", 
            random_state=self.random_state
        )
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step
        )

    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'RFESelector':
        total_features = len(dense_names) + len(sparse_names)
        logger.info(f"Fitting RFESelector (n_features_to_select={self.n_features_to_select}, total_features={total_features})...")
        
        # 1. Scaling dense features for Chi-Square pre-filter
        X_dense_scaled = self.scaler.fit_transform(X_dense) if len(dense_names) > 0 else np.empty((len(y), 0))
        
        # Combine dense and sparse features into a single sparse matrix for unified operations
        if len(dense_names) > 0:
            X_dense_sparse = scipy.sparse.csr_matrix(X_dense_scaled)
            if X_sparse.shape[1] > 0:
                X_all = scipy.sparse.hstack([X_dense_sparse, X_sparse]).tocsr()
            else:
                X_all = X_dense_sparse
        else:
            X_all = X_sparse.tocsr()
            
        all_names = dense_names + sparse_names
        
        # Initialize default rankings: features not in pre-selected pool will have a low rank (high rank number)
        default_rank_val = float(total_features + 1)
        for name in all_names:
            self.feature_rankings[name] = default_rank_val

        # 2. Chi-Square Pre-Filtering if feature space is large
        if total_features > self.pre_selected_pool_size:
            logger.info(f"Reducing feature space from {total_features} to {self.pre_selected_pool_size} using Chi-Square pre-filter for RFE...")
            chi2_scores, _ = chi2(X_all, y)
            chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
            
            top_indices = np.argsort(chi2_scores)[::-1][:self.pre_selected_pool_size]
            
            X_rfe = X_all[:, top_indices].toarray()  # Convert to dense for RFE
            rfe_feature_names = [all_names[idx] for idx in top_indices]
        else:
            X_rfe = X_all.toarray()
            rfe_feature_names = all_names
            top_indices = np.arange(total_features)

        # 3. Fit RFE
        logger.info(f"Running Recursive Feature Elimination (RFE) on {X_rfe.shape[1]} features...")
        # Scale sliced features to ensure Logistic Regression coefficients are stable
        X_rfe_scaled = self.scaler.fit_transform(X_rfe)
        self.rfe.fit(X_rfe_scaled, y)
        
        # 4. Extract rankings and support
        rfe_support = self.rfe.get_support()
        rfe_ranks = self.rfe.ranking_
        
        # Map back to original names
        for name, rank, keep in zip(rfe_feature_names, rfe_ranks, rfe_support):
            # In RFE: 1 is the best. Let's keep it as is (1 is best/selected)
            self.feature_rankings[name] = float(rank)
            if keep:
                self.selected_features.append(name)
                
        logger.info(f"RFESelector completed: selected {len(self.selected_features)} features.")
        return self

    def get_selected_features(self) -> List[str]:
        return self.selected_features

    def get_feature_rankings(self) -> Dict[str, float]:
        return self.feature_rankings

