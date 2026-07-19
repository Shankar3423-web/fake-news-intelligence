import os
import json
import pytest
from ml.training.training_config import TrainingConfig
from ml.training.version_manager import VersionManager

def test_version_manager_run(tmp_path):
    config_yaml = tmp_path / "training_config.yaml"
    with open(config_yaml, "w") as f:
        f.write(f"""
outputs:
  versions_file: "{tmp_path.as_posix()}/training_versions.json"
""")
        
    config = TrainingConfig(str(config_yaml))
    manager = VersionManager(config)
    
    # Assert initially empty list
    history = manager.load_versions()
    assert len(history) == 0
    
    # Register run 1
    entry = manager.register_run(
        dataset_version="1.0.1",
        feature_selection_version="2",
        model_ids=["xgb_1"],
        pipeline_hash="abcd1234hash",
        files_dict={}
    )
    
    assert entry["version"] == 1
    assert entry["dataset_version"] == "1.0.1"
    assert entry["feature_selection_version"] == "2"
    assert "timestamp" in entry
    
    # Reload and register run 2
    history2 = manager.load_versions()
    assert len(history2) == 1
    
    entry2 = manager.register_run(
        dataset_version="1.0.1",
        feature_selection_version="2",
        model_ids=["xgb_2"],
        pipeline_hash="efgh5678hash",
        files_dict={}
    )
    assert entry2["version"] == 2
