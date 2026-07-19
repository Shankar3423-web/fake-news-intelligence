import os
import json
import pytest
import pandas as pd
from ml.training.dataset_loader import DatasetLoader

def test_dataset_loader_success(tmp_path):
    # Create temporary dataset and features files
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    # Write mock features list
    feature_names = ["feat1", "feat2"]
    with open(features_file, "w") as f:
        json.dump(feature_names, f)
        
    # Write mock dataset
    df = pd.DataFrame({
        "id": list(range(1, 11)),
        "label": [0]*5 + [1]*5,
        "cleaned_text": [f"text{i}" for i in range(1, 11)],
        "feat1": [0.1 * i for i in range(1, 11)],
        "feat2": [0.2 * i for i in range(1, 11)]
    })
    df.to_csv(dataset_file, index=False)
    
    loader = DatasetLoader(str(dataset_file), str(features_file))
    df_loaded, loaded_feats = loader.load_and_validate()
    
    assert len(df_loaded) == 10
    assert loaded_feats == feature_names
    assert "cleaned_text" in df_loaded.columns

def test_dataset_loader_missing_files():
    loader = DatasetLoader("non_existent_dataset.csv", "non_existent_features.json")
    with pytest.raises(FileNotFoundError):
        loader.load_and_validate()

def test_dataset_loader_missing_base_column(tmp_path):
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    with open(features_file, "w") as f:
        json.dump(["feat1"], f)
        
    # Missing 'label'
    df = pd.DataFrame({
        "id": [1, 2],
        "cleaned_text": ["text1", "text2"],
        "feat1": [0.1, 0.2]
    })
    df.to_csv(dataset_file, index=False)
    
    loader = DatasetLoader(str(dataset_file), str(features_file))
    with pytest.raises(ValueError, match="Required base column 'label' is missing"):
        loader.load_and_validate()

def test_dataset_loader_non_numeric_features(tmp_path):
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    with open(features_file, "w") as f:
        json.dump(["feat1"], f)
        
    df = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 1],
        "cleaned_text": ["text1", "text2"],
        "feat1": ["not_numeric", "not_numeric"]
    })
    df.to_csv(dataset_file, index=False)
    
    loader = DatasetLoader(str(dataset_file), str(features_file))
    with pytest.raises(ValueError, match="Features must be of numeric type"):
        loader.load_and_validate()

def test_dataset_loader_invalid_labels(tmp_path):
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    with open(features_file, "w") as f:
        json.dump(["feat1"], f)
        
    # Labels contains 2 (invalid)
    df = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 2],
        "cleaned_text": ["text1", "text2"],
        "feat1": [0.1, 0.2]
    })
    df.to_csv(dataset_file, index=False)
    
    loader = DatasetLoader(str(dataset_file), str(features_file))
    with pytest.raises(ValueError, match="Label column contains invalid values"):
        loader.load_and_validate()
