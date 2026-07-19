import os
import pytest
from ml.training.training_config import TrainingConfig
from ml.training.training_pipeline import TrainingPipeline

def test_pipeline_instantiation(tmp_path):
    config_yaml = tmp_path / "training_config.yaml"
    
    config_content = f"""
random_state: 42
split:
  test_size: 0.2
  stratify: true
  shuffle: true
inputs:
  dataset_csv: "{tmp_path.as_posix()}/selected_feature_dataset_v1.csv"
  feature_names_json: "{tmp_path.as_posix()}/selected_feature_names.json"
  selection_versions_json: "{tmp_path.as_posix()}/selection_versions.json"
outputs:
  models_dir: "{tmp_path.as_posix()}/models"
  metadata_dir: "{tmp_path.as_posix()}/metadata"
  reports_dir: "{tmp_path.as_posix()}/reports"
  statistics_dir: "{tmp_path.as_posix()}/statistics"
  versions_dir: "{tmp_path.as_posix()}/versions"
  hashes_dir: "{tmp_path.as_posix()}/hashes"
  logs_dir: "{tmp_path.as_posix()}/logs"
  registry_file: "{tmp_path.as_posix()}/registry.json"
  statistics_file: "{tmp_path.as_posix()}/training_statistics.json"
  report_file: "{tmp_path.as_posix()}/training_report.md"
  versions_file: "{tmp_path.as_posix()}/training_versions.json"
models:
  logistic_regression:
    enabled: true
    hyperparameters:
      solver: "lbfgs"
  xgboost:
    enabled: false
"""
    with open(config_yaml, "w") as f:
        f.write(config_content)
        
    pipeline = TrainingPipeline(str(config_yaml))
    assert pipeline.config.random_state == 42
    assert pipeline.config.is_model_enabled("logistic_regression") is True
    assert pipeline.config.is_model_enabled("xgboost") is False
