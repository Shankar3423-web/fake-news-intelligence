import os
import json
import pytest
import joblib
from ml.evaluation.model_loader import ModelLoader

class DummyModel:
    def predict(self, X):
        return [0] * len(X)

def test_model_loader_missing_files(tmp_path):
    loader = ModelLoader(models_root_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load_model("logistic_regression", ["feat1"])

def test_model_loader_success(tmp_path):
    model_dir = tmp_path / "logistic_regression"
    model_dir.mkdir()
    
    # Save dummy model, config, feature order, and metadata
    dummy_model = DummyModel()
    joblib.dump(dummy_model, model_dir / "model.joblib")
    
    with open(model_dir / "feature_order.json", "w") as f:
        json.dump(["feat1", "feat2"], f)
        
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({"model_name": "lr_test"}, f)
        
    with open(model_dir / "training_config.json", "w") as f:
        json.dump({}, f)
        
    loader = ModelLoader(models_root_dir=str(tmp_path))
    model, metadata, feature_order = loader.load_model("logistic_regression", ["feat1", "feat2"])
    
    assert metadata["model_name"] == "lr_test"
    assert feature_order == ["feat1", "feat2"]
    assert model is not None

def test_model_loader_incompatible_features(tmp_path):
    model_dir = tmp_path / "logistic_regression"
    model_dir.mkdir()
    
    dummy_model = DummyModel()
    joblib.dump(dummy_model, model_dir / "model.joblib")
    
    with open(model_dir / "feature_order.json", "w") as f:
        json.dump(["feat1", "feat2"], f)
        
    with open(model_dir / "metadata.json", "w") as f:
        json.dump({}, f)
        
    with open(model_dir / "training_config.json", "w") as f:
        json.dump({}, f)
        
    loader = ModelLoader(models_root_dir=str(tmp_path))
    
    # Mismatched order/names
    with pytest.raises(ValueError, match="Feature ordering mismatch"):
        loader.load_model("logistic_regression", ["feat2", "feat1"])
        
    # Mismatched count
    with pytest.raises(ValueError, match="Feature count mismatch"):
        loader.load_model("logistic_regression", ["feat1"])
