import os
import json
import pytest
import pandas as pd
from unittest.mock import patch
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_pipeline import EvaluationPipeline

import numpy as np

class MockModel:
    def predict(self, X):
        return np.array([0] * len(X))
    def predict_proba(self, X):
        return np.array([[0.9, 0.1]] * len(X))

@patch("ml.evaluation.model_loader.ModelLoader.load_model")
def test_evaluation_pipeline(mock_load_model, tmp_path):
    # Set up mock model loader responses
    mock_load_model.return_value = (
        MockModel(),
        {
            "model_name": "mock_model_123",
            "algorithm": "Mock Classifier",
            "path": "ml/training/models/mock",
            "dataset_version": "1.0.1",
            "feature_selection_version": "2"
        },
        ["feat1", "feat2"]
    )
    
    # Create temp dataset and feature names files
    dataset_file = tmp_path / "mock_dataset.csv"
    features_file = tmp_path / "mock_features.json"
    
    with open(features_file, "w") as f:
        json.dump(["feat1", "feat2"], f)
        
    df = pd.DataFrame({
        "id": list(range(1, 21)),
        "label": [0]*10 + [1]*10,
        "cleaned_text": [f"text{i}" for i in range(1, 21)],
        "feat1": [0.1 * i for i in range(1, 21)],
        "feat2": [0.2 * i for i in range(1, 21)]
    })
    df.to_csv(dataset_file, index=False)
    
    # Create config YAML
    config_yaml = tmp_path / "evaluation_config.yaml"
    with open(config_yaml, "w") as f:
        f.write(f"""
random_state: 42
split:
  test_size: 0.2
  stratify: true
  shuffle: true

inputs:
  dataset_csv: {str(dataset_file).replace("\\", "/")}
  feature_names_json: {str(features_file).replace("\\", "/")}
  training_registry_json: {str(tmp_path).replace("\\", "/")}  # won't be read in mocked load
  training_versions_json: {str(tmp_path).replace("\\", "/")}

outputs:
  reports_dir: {str(tmp_path / "reports").replace("\\", "/")}
  statistics_dir: {str(tmp_path / "statistics").replace("\\", "/")}
  metadata_dir: {str(tmp_path / "metadata").replace("\\", "/")}
  hashes_dir: {str(tmp_path / "hashes").replace("\\", "/")}
  versions_dir: {str(tmp_path / "versions").replace("\\", "/")}
  leaderboard_dir: {str(tmp_path / "leaderboard").replace("\\", "/")}
  comparison_dir: {str(tmp_path / "comparison").replace("\\", "/")}
  logs_dir: {str(tmp_path / "logs").replace("\\", "/")}
  predictions_dir: {str(tmp_path / "predictions").replace("\\", "/")}
  classification_reports_dir: {str(tmp_path / "classification_reports").replace("\\", "/")}
  confusion_matrices_dir: {str(tmp_path / "confusion_matrices").replace("\\", "/")}
  roc_curves_dir: {str(tmp_path / "roc_curves").replace("\\", "/")}
  precision_recall_curves_dir: {str(tmp_path / "precision_recall_curves").replace("\\", "/")}
  charts_dir: {str(tmp_path / "charts").replace("\\", "/")}
  
  evaluation_report_file: {str(tmp_path / "reports/evaluation_report.md").replace("\\", "/")}
  evaluation_statistics_file: {str(tmp_path / "statistics/evaluation_statistics.json").replace("\\", "/")}
  leaderboard_csv_file: {str(tmp_path / "leaderboard/leaderboard.csv").replace("\\", "/")}
  leaderboard_json_file: {str(tmp_path / "leaderboard/leaderboard.json").replace("\\", "/")}
  leaderboard_md_file: {str(tmp_path / "leaderboard/leaderboard.md").replace("\\", "/")}
  comparison_csv_file: {str(tmp_path / "comparison/model_comparison.csv").replace("\\", "/")}
  comparison_json_file: {str(tmp_path / "comparison/model_comparison.json").replace("\\", "/")}
  comparison_md_file: {str(tmp_path / "comparison/model_comparison.md").replace("\\", "/")}
  best_model_file: {str(tmp_path / "best_model.json").replace("\\", "/")}
  versions_file: {str(tmp_path / "versions/evaluation_versions.json").replace("\\", "/")}
  hashes_file: {str(tmp_path / "hashes/evaluation_hashes.json").replace("\\", "/")}

visualizations:
  enable_charts: true
  enable_roc: true
  enable_pr_curve: true

best_model_selection:
  selection_metric: weighted_score
  weights:
    f1_score: 0.40
    roc_auc: 0.30
    precision: 0.15
    recall: 0.10
    prediction_speed: 0.05
""")
        
    pipeline = EvaluationPipeline(str(config_yaml))
    success = pipeline.run()
    
    assert success is True
    assert os.path.exists(tmp_path / "best_model.json")
    assert os.path.exists(tmp_path / "reports/evaluation_report.md")
    assert os.path.exists(tmp_path / "leaderboard/leaderboard.json")
