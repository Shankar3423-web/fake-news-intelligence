import os
from ml.evaluation.report_generator import ReportGenerator

def test_report_generator(tmp_path):
    report_file = tmp_path / "evaluation_report.md"
    
    dataset_info = {
        "path": "data.csv",
        "size_bytes": 1024,
        "total_rows": 1000,
        "test_rows": 200,
        "test_size": 0.2,
        "feature_count": 50,
        "random_state": 42
    }
    
    training_info = {
        "training_version": "training_v1",
        "dataset_version": "1.0.1",
        "feature_selection_version": "2"
    }
    
    model_results = {
        "xgboost": {
            "algorithm": "XGBoost",
            "model_id": "model_xgb_123",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.96,
                "f1_score": 0.95,
                "roc_auc": 0.97,
                "prediction_time_sec": 0.05,
                "inference_throughput_sps": 200.0,
                "memory_used_mb": 4.5,
                "model_size_mb": 0.02
            }
        }
    }
    
    overall_scores = {"xgboost": 0.95}
    leaderboard = [{
        "Rank": 1,
        "Model": "XGBoost",
        "Model Key": "xgboost",
        "Overall Score": 0.95,
        "F1": 0.95,
        "ROC": 0.97,
        "Prediction Speed": 200.0,
        "Memory": 4.5
    }]
    best_model = {
        "algorithm": "XGBoost",
        "model_key": "xgboost",
        "model_id": "model_xgb_123",
        "overall_score": 0.95,
        "selection_metric_used": "weighted_score",
        "metrics": {"model_size_mb": 0.02, "inference_latency_ms": 0.5}
    }
    
    generator = ReportGenerator(str(report_file))
    generator.generate(
        dataset_info=dataset_info,
        training_info=training_info,
        model_results=model_results,
        overall_scores=overall_scores,
        leaderboard=leaderboard,
        best_model=best_model,
        pipeline_duration=2.5,
        generated_files={"leaderboard_csv": "lead.csv"},
        warnings=[]
    )
    
    assert os.path.exists(report_file)
    with open(report_file, "r", encoding="utf-8") as f:
        text = f.read()
    assert "Phase 7: Model Evaluation Report" in text
    assert "XGBoost" in text
