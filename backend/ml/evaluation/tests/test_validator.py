import os
import pytest
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.evaluation_validator import EvaluationValidator

def test_validator_fails_when_files_missing(tmp_path):
    config_yaml = tmp_path / "evaluation_config.yaml"
    with open(config_yaml, "w") as f:
        f.write(f"""
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
""")
        
    config = EvaluationConfig(str(config_yaml))
    validator = EvaluationValidator(config)
    
    # Since directories and files are missing, it should fail
    success, errors = validator.validate_all()
    assert success is False
    assert len(errors) > 0
