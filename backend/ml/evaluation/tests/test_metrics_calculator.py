import numpy as np
from ml.evaluation.metrics_calculator import MetricsCalculator

def test_metrics_calculator_success():
    calc = MetricsCalculator()
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.9])
    
    metrics = calc.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        y_prob_type="predict_proba",
        duration_sec=1.5,
        throughput_sps=4.0,
        memory_used_mb=12.5,
        model_size_bytes=1048576 # 1 MB
    )
    
    assert metrics["accuracy"] == pytest_approx(5/6, 1e-4)
    assert metrics["f1_score"] > 0.0
    assert metrics["roc_auc"] > 0.0
    assert metrics["log_loss"] is not None
    assert metrics["model_size_mb"] == 1.0

def test_metrics_calculator_no_probs():
    calc = MetricsCalculator()
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    metrics = calc.calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=None,
        y_prob_type="none",
        duration_sec=0.1,
        throughput_sps=40.0,
        memory_used_mb=1.5,
        model_size_bytes=1024
    )
    
    assert metrics["accuracy"] == 1.0
    assert metrics["log_loss"] is None

def pytest_approx(expected, tolerance):
    return pytest_approx_obj(expected, tolerance)

class pytest_approx_obj:
    def __init__(self, expected, tolerance):
        self.expected = expected
        self.tolerance = tolerance
    def __eq__(self, actual):
        return abs(actual - self.expected) <= self.tolerance
