import os
import json
import pytest
import pandas as pd
from ml.evaluation.dataset_loader import DatasetLoader
from ml.evaluation.evaluation_config import EvaluationConfig

def test_dataset_loader_success(tmp_path):
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    feature_names = ["feat1", "feat2"]
    with open(features_file, "w") as f:
        json.dump(feature_names, f)
        
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

def test_dataset_loader_split(tmp_path):
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    feature_names = ["feat1", "feat2"]
    with open(features_file, "w") as f:
        json.dump(feature_names, f)
        
    df = pd.DataFrame({
        "id": list(range(1, 21)),
        "label": [0]*10 + [1]*10,
        "cleaned_text": [f"text{i}" for i in range(1, 21)],
        "feat1": [0.1 * i for i in range(1, 21)],
        "feat2": [0.2 * i for i in range(1, 21)]
    })
    df.to_csv(dataset_file, index=False)
    
    config = EvaluationConfig(config_path="non_existent.yaml")
    # Config default: test_size = 0.2
    
    loader = DatasetLoader(str(dataset_file), str(features_file))
    X_test, y_test = loader.load_test_split(config)
    
    assert X_test.shape == (4, 2)
    assert len(y_test) == 4
