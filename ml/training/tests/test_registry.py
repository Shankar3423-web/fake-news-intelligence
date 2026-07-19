import os
import json
import pytest
from ml.training.model_registry import ModelRegistry

def test_model_registry_lifecycle(tmp_path):
    registry_file = tmp_path / "registry.json"
    
    registry = ModelRegistry(str(registry_file))
    initial = registry.load_registry()
    assert initial["latest_model_id"] == ""
    assert len(initial["models"]) == 0
    
    # Register a model
    registry.register_model(
        model_id="xgb_run_1",
        algorithm="XGBoost",
        version="training_v1",
        dataset_version="1.0.1",
        training_date="2026-07-19T00:00:00",
        feature_count=100,
        training_samples=1000,
        testing_samples=250,
        path="ml/training/models/xgboost"
    )
    
    # Reload and verify
    reloaded = registry.load_registry()
    assert reloaded["latest_model_id"] == "xgb_run_1"
    assert len(reloaded["models"]) == 1
    
    entry = registry.get_model_entry("xgb_run_1")
    assert entry is not None
    assert entry["algorithm"] == "XGBoost"
    assert entry["feature_count"] == 100
    
    # Overwrite registry with update
    registry.register_model(
        model_id="xgb_run_1",
        algorithm="XGBoost",
        version="training_v1.1",
        dataset_version="1.0.1",
        training_date="2026-07-19T01:00:00",
        feature_count=100,
        training_samples=1000,
        testing_samples=250,
        path="ml/training/models/xgboost"
    )
    
    updated = registry.load_registry()
    assert len(updated["models"]) == 1
    assert registry.get_model_entry("xgb_run_1")["version"] == "training_v1.1"
