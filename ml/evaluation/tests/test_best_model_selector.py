import os
import json
import pytest
from ml.evaluation.evaluation_config import EvaluationConfig
from ml.evaluation.best_model_selector import BestModelSelector

def test_best_model_selector_weighted(tmp_path):
    # Create temp config YAML
    config_yaml = tmp_path / "evaluation_config.yaml"
    best_model_file = tmp_path / "best_model.json"
    
    with open(config_yaml, "w") as f:
        f.write(f"""
best_model_selection:
  selection_metric: weighted_score
  weights:
    f1_score: 0.40
    roc_auc: 0.30
    precision: 0.15
    recall: 0.10
    prediction_speed: 0.05
outputs:
  best_model_file: {str(best_model_file).replace("\\", "/")}
""")
        
    config = EvaluationConfig(str(config_yaml))
    selector = BestModelSelector(config)
    
    model_eval_results = {
        "xgboost": {
            "algorithm": "XGBoost",
            "model_id": "model_xgboost_123",
            "metadata": {"path": "ml/models/xgb"},
            "metrics": {
                "f1_score": 0.95,
                "roc_auc": 0.97,
                "precision": 0.94,
                "recall": 0.96,
                "inference_throughput_sps": 200.0,
                "model_size_mb": 1.5,
                "inference_latency_ms": 0.5
            },
            "predictions": {"y_pred": [0, 1]}
        },
        "logistic_regression": {
            "algorithm": "Logistic Regression",
            "model_id": "model_lr_123",
            "metadata": {"path": "ml/models/lr"},
            "metrics": {
                "f1_score": 0.90,
                "roc_auc": 0.92,
                "precision": 0.89,
                "recall": 0.91,
                "inference_throughput_sps": 1000.0,
                "model_size_mb": 0.1,
                "inference_latency_ms": 0.1
            },
            "predictions": {"y_pred": [0, 1]}
        }
    }
    
    scores = selector.calculate_overall_scores(model_eval_results)
    best_key, best_data = selector.select_best_model(model_eval_results, scores)
    
    assert best_key == "xgboost"
    assert best_data["model_id"] == "model_xgboost_123"
    assert os.path.exists(best_model_file)
    
    with open(best_model_file, "r") as f:
        saved = json.load(f)
    assert saved["model_key"] == "xgboost"
