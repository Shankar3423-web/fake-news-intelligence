import os
import json
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.version_manager import VersionManager

def test_version_manager(tmp_path):
    config_yaml = tmp_path / "evaluation_config.yaml"
    versions_file = tmp_path / "evaluation_versions.json"
    
    with open(config_yaml, "w") as f:
        f.write(f"""
outputs:
  versions_file: {str(versions_file).replace("\\", "/")}
""")
        
    config = EvaluationConfig(str(config_yaml))
    manager = VersionManager(config)
    
    assert manager.load_versions() == []
    
    # Register run
    manager.register_run(
        training_version="training_v2",
        dataset_version="1.0.1",
        feature_selection_version="2",
        best_model_id="model_xgboost_123",
        pipeline_hash="fake_hash_12345",
        files_dict={}
    )
    
    runs = manager.load_versions()
    assert len(runs) == 1
    assert runs[0]["training_version"] == "training_v2"
    assert runs[0]["evaluation_version"] == "evaluation_v1"
    assert runs[0]["best_model_id"] == "model_xgboost_123"
