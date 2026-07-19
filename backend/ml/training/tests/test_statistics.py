import os
import json
import pytest
from ml.training.training_statistics import TrainingStatistics

def test_training_statistics_generation(tmp_path):
    stats_file = tmp_path / "training_statistics.json"
    dataset_file = tmp_path / "dataset.csv"
    
    # Create fake dataset to measure size
    with open(dataset_file, "w") as f:
        f.write("id,label,feat1,feat2\n1,0,0.1,0.2\n")
        
    stats = TrainingStatistics()
    summary = stats.generate(
        dataset_path=str(dataset_file),
        feature_count=2,
        training_rows=8,
        testing_rows=2,
        total_pipeline_duration=12.34,
        peak_memory_used_mb=150.5,
        model_durations={"logistic_regression": 5.1, "svm": 7.24}
    )
    
    assert summary["feature_count"] == 2
    assert summary["training_rows"] == 8
    assert summary["average_training_time_sec"] == 6.17
    assert summary["total_models_trained"] == 2
    
    # Save and verify load
    stats.save(str(stats_file))
    assert os.path.exists(stats_file)
    
    with open(stats_file, "r") as f:
        loaded = json.load(f)
        
    assert loaded["average_training_time_sec"] == 6.17
    assert loaded["dataset_size_bytes"] > 0
