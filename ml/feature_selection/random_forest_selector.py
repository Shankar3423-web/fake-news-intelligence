import logging
import pandas as pd
import numpy as np
import scipy.sparse
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from ml.feature_selection.selection_utils import BaseFeatureSelector

logger = logging.getLogger("feature_selection_pipeline")

class RandomForestSelector(BaseFeatureSelector):
    """
    Selects top features using the feature importances of a Random Forest Classifier.
    To ensure speed on high-dimensional spaces, a pre-filtering step (using Chi-Square)
    reduces the feature space to a pre_selected_pool_size before training the Random Forest.
    """
    def __init__(
        self, 
        top_k: int = 100,
        n_estimators: int = 50,
        max_depth: int = 10,
        random_state: int = 42,
        pre_selected_pool_size: int = 300
    ) -> None:
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.pre_selected_pool_size = pre_selected_pool_size
        self.selected_features: List[str] = []
        self.feature_rankings: Dict[str, float] = {}
        self.scaler = MinMaxScaler()
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores for speed
        )

    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'RandomForestSelector':
        total_features = len(dense_names) + len(sparse_names)
        logger.info(f"Fitting RandomForestSelector (top_k={self.top_k}, total_features={total_features})...")
        
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
        
        # Set all initial rankings to 0.0
        for name in all_names:
            self.feature_rankings[name] = 0.0

        # 2. Chi-Square Pre-Filtering if feature space is large
        if total_features > self.pre_selected_pool_size:
            logger.info(f"Reducing feature space from {total_features} to {self.pre_selected_pool_size} using Chi-Square pre-filter...")
            chi2_scores, _ = chi2(X_all, y)
            # Handle NaNs in chi2 scores
            chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
            
            # Get indices of top features
            top_indices = np.argsort(chi2_scores)[::-1][:self.pre_selected_pool_size]
            
            # Slicing the matrix
            X_rf = X_all[:, top_indices].toarray()  # Convert to dense for Random Forest
            rf_feature_names = [all_names[idx] for idx in top_indices]
        else:
            X_rf = X_all.toarray()
            rf_feature_names = all_names
            top_indices = np.arange(total_features)

        # 3. Train Random Forest Classifier
        logger.info(f"Training Random Forest on {X_rf.shape[1]} features...")
        self.rf.fit(X_rf, y)
        
        # 4. Extract importances
        importances = self.rf.feature_importances_
        
        # Map importances back to original names
        rf_rankings = {}
        for name, imp in zip(rf_feature_names, importances):
            self.feature_rankings[name] = float(imp)
            rf_rankings[name] = float(imp)
            
        # Select top K features
        sorted_features = sorted(rf_rankings.keys(), key=lambda k: rf_rankings[k], reverse=True)
        k_sel = min(self.top_k, len(sorted_features))
        self.selected_features = sorted_features[:k_sel]
        
        logger.info(f"RandomForestSelector completed: selected {len(self.selected_features)} features.")
        return self

    def get_selected_features(self) -> List[str]:
        return self.selected_features

    def get_feature_rankings(self) -> Dict[str, float]:
        return self.feature_rankings
