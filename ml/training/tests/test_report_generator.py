import os
import pytest
from ml.training.report_generator import ReportGenerator

def test_report_generator(tmp_path):
    report_file = tmp_path / "training_report.md"
    
    gen = ReportGenerator(str(report_file))
    
    dataset_info = {"path": "dense.csv", "size_bytes": 1024, "feature_count": 10}
    split_info = {
        "training_samples": 80,
        "testing_samples": 20,
        "label_distribution": {
            "train": {0: 40, 1: 40},
            "test": {0: 10, 1: 10}
        }
    }
    model_summaries = [
        {
            "algorithm": "Logistic Regression",
            "training_duration_sec": 1.2,
            "memory_used_mb": 10.0,
            "samples_trained": 80,
            "feature_count": 10,
            "hyperparameters": {"C": 1.0}
        }
    ]
    generated_files = {"model": "model.joblib"}
    
    md_content = gen.generate(
        dataset_info=dataset_info,
        split_info=split_info,
        model_summaries=model_summaries,
        pipeline_duration=2.5,
        generated_files=generated_files,
        warnings=["Imbalanced classes test warning"]
    )
    
    assert os.path.exists(report_file)
    assert "# Phase 6" in md_content
    assert "Imbalanced classes test warning" in md_content
    assert "Logistic Regression" in md_content
