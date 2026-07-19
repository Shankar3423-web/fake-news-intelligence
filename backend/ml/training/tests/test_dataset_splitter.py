import pytest
import pandas as pd
from ml.training.dataset_splitter import DatasetSplitter

def test_dataset_splitter_success():
    df = pd.DataFrame({
        "id": list(range(10)),
        "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "feat1": [0.1 * i for i in range(10)],
        "feat2": [0.2 * i for i in range(10)]
    })
    
    splitter = DatasetSplitter(test_size=0.2, random_state=42, stratify=True)
    X_train, X_test, y_train, y_test = splitter.split(df, ["feat1", "feat2"])
    
    # Check dimensions
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    
    # Check stratification (equal split ratio 1:1, so train/test should be equal distribution)
    assert y_train.value_counts().to_dict() == {0: 4, 1: 4}
    assert y_test.value_counts().to_dict() == {0: 1, 1: 1}
    
    # Verify split metadata
    info = splitter.get_split_info()
    assert info["training_samples"] == 8
    assert info["testing_samples"] == 2
    assert info["config"]["test_size"] == 0.2
    assert info["config"]["stratify"] is True

def test_dataset_splitter_unsplit_error():
    splitter = DatasetSplitter()
    with pytest.raises(ValueError, match="Dataset has not been split yet"):
        splitter.get_split_info()
