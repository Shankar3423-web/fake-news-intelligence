import numpy as np
import pandas as pd
import scipy.sparse
import pytest
from ml.feature_selection.variance_selector import VarianceSelector
from ml.feature_selection.correlation_selector import CorrelationSelector
from ml.feature_selection.mutual_information_selector import MutualInformationSelector
from ml.feature_selection.chi_square_selector import ChiSquareSelector
from ml.feature_selection.random_forest_selector import RandomForestSelector
from ml.feature_selection.rfe_selector import RFESelector

@pytest.fixture
def dummy_data() -> tuple[pd.DataFrame, scipy.sparse.csr_matrix, np.ndarray, list[str], list[str]]:
    # Create 100 samples
    np.random.seed(42)
    y = np.random.choice([0, 1], size=100)
    
    # Dense features:
    # d1: constant (zero variance)
    # d2: high correlation with y
    # d3: highly correlated with d2
    # d4: standard random
    d1 = np.zeros(100)
    d2 = y * 2.0 + np.random.normal(0, 0.1, 100)
    d3 = d2 + np.random.normal(0, 0.01, 100)  # highly correlated with d2
    d4 = np.random.normal(0, 1, 100)
    
    X_dense = pd.DataFrame({
        "dense_const": d1,
        "dense_pred": d2,
        "dense_corr": d3,
        "dense_rand": d4
    })
    
    # Sparse features (TF-IDF): non-negative
    # s1: constant (zero variance)
    # s2: predictive
    # s3: random
    s1 = np.ones((100, 1)) * 0.5
    s2 = (y[:, np.newaxis] * 0.8 + np.random.uniform(0, 0.2, (100, 1)))
    s3 = np.random.uniform(0, 1, (100, 5))
    
    X_sparse = scipy.sparse.csr_matrix(np.hstack([s1, s2, s3]))
    
    dense_names = list(X_dense.columns)
    sparse_names = [f"tfidf_feat_{i}" for i in range(7)]
    
    return X_dense, X_sparse, y, dense_names, sparse_names

def test_variance_selector(dummy_data) -> None:
    X_dense, X_sparse, y, dense_names, sparse_names = dummy_data
    
    # Set threshold to keep features with variance > 0.01
    selector = VarianceSelector(threshold=0.01)
    selector.fit(X_dense, X_sparse, y, dense_names, sparse_names)
    
    selected = selector.get_selected_features()
    rankings = selector.get_feature_rankings()
    
    # dense_const and tfidf_feat_0 have zero variance, so they should be filtered out
    assert "dense_const" not in selected
    assert "tfidf_feat_0" not in selected
    
    # Make sure rankings are populated
    assert "dense_pred" in rankings
    assert rankings["dense_const"] < 0.0001

def test_correlation_selector(dummy_data) -> None:
    X_dense, X_sparse, y, dense_names, sparse_names = dummy_data
    
    # correlation threshold = 0.8
    selector = CorrelationSelector(threshold=0.8)
    selector.fit(X_dense, X_sparse, y, dense_names, sparse_names)
    
    selected = selector.get_selected_features()
    rankings = selector.get_feature_rankings()
    
    # Either dense_pred or dense_corr should be dropped because they are highly correlated (r > 0.99)
    # Since dense_pred is slightly more correlated with y, dense_corr should be dropped
    assert "dense_pred" in selected
    assert "dense_corr" not in selected
    assert "tfidf_feat_0" in selected  # Sparse features kept by correlation selector

def test_mutual_information_selector(dummy_data) -> None:
    X_dense, X_sparse, y, dense_names, sparse_names = dummy_data
    
    selector = MutualInformationSelector(top_k_dense=2, top_k_sparse=2, sub_sample_size=1000)
    selector.fit(X_dense, X_sparse, y, dense_names, sparse_names)
    
    selected = selector.get_selected_features()
    
    # Verify selected feature counts
    dense_sel = [f for f in selected if f in dense_names]
    sparse_sel = [f for f in selected if f in sparse_names]
    
    assert len(dense_sel) <= 2
    assert len(sparse_sel) <= 2
    # The predictive dense feature should be selected
    assert "dense_pred" in dense_sel

def test_chi_square_selector(dummy_data) -> None:
    X_dense, X_sparse, y, dense_names, sparse_names = dummy_data
    
    selector = ChiSquareSelector(top_k_dense=2, top_k_sparse=2)
    selector.fit(X_dense, X_sparse, y, dense_names, sparse_names)
    
    selected = selector.get_selected_features()
    
    dense_sel = [f for f in selected if f in dense_names]
    sparse_sel = [f for f in selected if f in sparse_names]
    
    assert len(dense_sel) <= 2
    assert len(sparse_sel) <= 2
    assert "dense_pred" in dense_sel

def test_random_forest_selector(dummy_data) -> None:
    X_dense, X_sparse, y, dense_names, sparse_names = dummy_data
    
    selector = RandomForestSelector(top_k=3, n_estimators=10, max_depth=5, pre_selected_pool_size=10)
    selector.fit(X_dense, X_sparse, y, dense_names, sparse_names)
    
    selected = selector.get_selected_features()
    rankings = selector.get_feature_rankings()
    
    assert len(selected) <= 3
    assert "dense_pred" in selected or "dense_corr" in selected
    assert all(name in rankings for name in dense_names + sparse_names)

def test_rfe_selector(dummy_data) -> None:
    X_dense, X_sparse, y, dense_names, sparse_names = dummy_data
    
    selector = RFESelector(n_features_to_select=3, step=2, pre_selected_pool_size=8)
    selector.fit(X_dense, X_sparse, y, dense_names, sparse_names)
    
    selected = selector.get_selected_features()
    
    assert len(selected) <= 3
    assert "dense_pred" in selected or "dense_corr" in selected
