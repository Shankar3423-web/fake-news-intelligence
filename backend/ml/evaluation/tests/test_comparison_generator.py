import os
import pytest
from ml.evaluation.comparison_generator import ComparisonGenerator

def test_comparison_generator(tmp_path):
    comp_dir = tmp_path / "comparison"
    charts_dir = tmp_path / "charts"
    
    comp_dir.mkdir()
    charts_dir.mkdir()
    
    model_eval_results = {
        "xgboost": {
            "algorithm": "XGBoost",
            "model_id": "model_xgboost_123",
            "model_size_bytes": 2048,
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "roc_auc": 0.97,
                "balanced_accuracy": 0.95,
                "mcc": 0.90,
                "cohen_kappa": 0.90,
                "log_loss": 0.12,
                "prediction_time_sec": 0.05,
                "inference_throughput_sps": 200.0,
                "memory_used_mb": 4.5,
                "model_size_mb": 0.02
            }
        },
        "logistic_regression": {
            "algorithm": "Logistic Regression",
            "model_id": "model_lr_123",
            "model_size_bytes": 1024,
            "metrics": {
                "accuracy": 0.90,
                "precision": 0.89,
                "recall": 0.91,
                "f1_score": 0.90,
                "roc_auc": 0.92,
                "balanced_accuracy": 0.90,
                "mcc": 0.80,
                "cohen_kappa": 0.80,
                "log_loss": 0.25,
                "prediction_time_sec": 0.01,
                "inference_throughput_sps": 1000.0,
                "memory_used_mb": 0.5,
                "model_size_mb": 0.01
            }
        }
    }
    
    overall_scores = {
        "xgboost": 0.945,
        "logistic_regression": 0.905
    }
    
    gen = ComparisonGenerator(comparison_dir=str(comp_dir), charts_dir=str(charts_dir))
    out = gen.generate(model_eval_results, overall_scores)
    
    assert os.path.exists(out["csv_path"])
    assert os.path.exists(out["json_path"])
    assert os.path.exists(out["md_path"])
    
    # Check if charts are created
    assert os.path.exists(os.path.join(charts_dir, "metrics_comparison.png"))
    assert os.path.exists(os.path.join(charts_dir, "prediction_time_comparison.png"))
    assert os.path.exists(os.path.join(charts_dir, "model_size_comparison.png"))
