import os
import json
import pytest
from ml.training.training_config import TrainingConfig
from ml.training.training_validator import TrainingValidator

def test_validator_fails_on_empty(tmp_path):
    config_yaml = tmp_path / "training_config.yaml"
    
    # Write a config with temp paths to trigger validation failures
    config_data = f"""
random_state: 42
split:
  test_size: 0.2
  stratify: true
  shuffle: true
inputs:
  dataset_csv: "{tmp_path.as_posix()}/non_existent.csv"
  feature_names_json: "{tmp_path.as_posix()}/non_existent.json"
  selection_versions_json: "{tmp_path.as_posix()}/non_existent.json"
outputs:
  models_dir: "{tmp_path.as_posix()}/models"
  metadata_dir: "{tmp_path.as_posix()}/metadata"
  reports_dir: "{tmp_path.as_posix()}/reports"
  statistics_dir: "{tmp_path.as_posix()}/statistics"
  versions_dir: "{tmp_path.as_posix()}/versions"
  hashes_dir: "{tmp_path.as_posix()}/hashes"
  logs_dir: "{tmp_path.as_posix()}/logs"
  registry_file: "{tmp_path.as_posix()}/registry.json"
  statistics_file: "{tmp_path.as_posix()}/statistics.json"
  report_file: "{tmp_path.as_posix()}/report.md"
  versions_file: "{tmp_path.as_posix()}/versions.json"
models:
  logistic_regression:
    enabled: true
    hyperparameters:
      C: 1.0
"""
    with open(config_yaml, "w") as f:
        f.write(config_data)
        
    config = TrainingConfig(str(config_yaml))
    validator = TrainingValidator(config)
    
    success, errors = validator.validate_all()
    assert success is False
    assert len(errors) > 0
    # Checks that it caught missing directories
    assert any("Output directory missing" in err for err in errors)
