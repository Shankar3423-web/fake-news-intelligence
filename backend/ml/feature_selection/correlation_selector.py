import logging
import pandas as pd
import numpy as np
import scipy.sparse
from typing import List, Dict, Any, Set
from ml.feature_selection.selection_utils import BaseFeatureSelector

logger = logging.getLogger("feature_selection_pipeline")

class CorrelationSelector(BaseFeatureSelector):
    """
    Selects features by removing highly correlated dense features.
    For pairs of features with correlation above the threshold, keeps the one 
    with the higher absolute correlation to the target variable y.
    """
    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self.selected_features: List[str] = []
        self.feature_rankings: Dict[str, float] = {}

    def fit(
        self, 
        X_dense: pd.DataFrame, 
        X_sparse: scipy.sparse.csr_matrix, 
        y: np.ndarray, 
        dense_names: List[str], 
        sparse_names: List[str]
    ) -> 'CorrelationSelector':
        logger.info(f"Fitting CorrelationSelector with threshold={self.threshold}...")
        
        # Sparse TF-IDF features are kept as-is since computing pairwise correlation
        # on 5,000 sparse columns is computationally prohibitive.
        self.selected_features = list(sparse_names)
        for name in sparse_names:
            self.feature_rankings[name] = 0.0  # TF-IDF not ranked by correlation selector
            
        if len(dense_names) == 0:
            logger.info("No dense features provided. CorrelationSelector completed.")
            return self

        # 1. Compute correlation of each dense feature with target y
        y_series = pd.Series(y)
        corr_with_y = {}
        for col in dense_names:
            try:
                # Compute pearson correlation with y
                c = X_dense[col].corr(y_series)
                corr_with_y[col] = float(np.abs(c)) if not pd.isna(c) else 0.0
            except Exception as e:
                logger.warning(f"Error computing correlation with y for {col}: {e}")
                corr_with_y[col] = 0.0
                
        # Fill rankings for dense features
        for col, val in corr_with_y.items():
            self.feature_rankings[col] = val

        # 2. Compute pairwise correlation of dense features
        corr_matrix = X_dense.corr(method="pearson").abs()
        
        # 3. Sort dense features by their correlation with y descending
        sorted_features = sorted(dense_names, key=lambda name: corr_with_y[name], reverse=True)
        
        dropped_features: Set[str] = set()
        kept_dense_features: List[str] = []
        
        for i, col in enumerate(sorted_features):
            if col in dropped_features:
                continue
                
            kept_dense_features.append(col)
            
            # Check all subsequent features for high correlation with this one
            for other_col in sorted_features[i + 1:]:
                if other_col in dropped_features:
                    continue
                    
                pairwise_corr = corr_matrix.loc[col, other_col]
                if not pd.isna(pairwise_corr) and pairwise_corr > self.threshold:
                    logger.debug(f"Feature '{other_col}' (r(y)={corr_with_y[other_col]:.4f}) is highly correlated with '{col}' (r(y)={corr_with_y[col]:.4f}, r={pairwise_corr:.4f}). Dropping '{other_col}'.")
                    dropped_features.add(other_col)
                    
        self.selected_features = kept_dense_features + self.selected_features
        logger.info(f"CorrelationSelector completed: selected {len(kept_dense_features)}/{len(dense_names)} dense features. Dropped {len(dropped_features)} redundant features.")
        return self

    def get_selected_features(self) -> List[str]:
        return self.selected_features

    def get_feature_rankings(self) -> Dict[str, float]:
        return self.feature_rankings
