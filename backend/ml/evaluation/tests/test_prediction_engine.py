import pandas as pd
import numpy as np
from ml.evaluation.prediction_engine import PredictionEngine

class DummyClassifier:
    def predict(self, X):
        return np.array([0] * len(X))

class DummyProbClassifier(DummyClassifier):
    def predict_proba(self, X):
        # returns array of [prob_class_0, prob_class_1]
        return np.array([[0.8, 0.2]] * len(X))

def test_prediction_engine_no_probs():
    model = DummyClassifier()
    engine = PredictionEngine(model)
    
    df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    out = engine.predict(df)
    
    assert np.array_equal(out["y_pred"], np.array([0, 0, 0]))
    assert out["y_prob"] is None
    assert out["y_prob_type"] == "none"
    assert out["prediction_duration_sec"] >= 0.0
    assert out["inference_throughput_sps"] >= 0.0
    assert out["inference_latency_ms"] >= 0.0

def test_prediction_engine_with_probs():
    model = DummyProbClassifier()
    engine = PredictionEngine(model)
    
    df = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    out = engine.predict(df)
    
    assert np.array_equal(out["y_pred"], np.array([0, 0, 0]))
    assert np.array_equal(out["y_prob"], np.array([0.2, 0.2, 0.2]))
    assert out["y_prob_type"] == "predict_proba"
    assert out["inference_throughput_sps"] >= 0.0
