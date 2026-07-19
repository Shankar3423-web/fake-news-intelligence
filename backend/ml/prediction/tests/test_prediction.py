import os
import tempfile
import json
import pytest
import pandas as pd
import numpy as np

from ml.prediction.input_validator import InputValidator
from ml.prediction.confidence_calculator import ConfidenceCalculator
from ml.prediction.response_builder import ResponseBuilder
from ml.prediction.prediction_statistics import PredictionStatistics
from ml.prediction.metadata_manager import MetadataManager
from ml.prediction.hash_generator import HashGenerator
from ml.prediction.version_manager import VersionManager
from ml.prediction.report_generator import ReportGenerator
from ml.prediction.prediction_validator import PredictionValidator
from ml.prediction.prediction_config import PredictionConfig

# 1. Test InputValidator
def test_input_validator():
    validator = InputValidator(min_length=10, max_length=100)
    
    # Valid text
    assert validator.validate("This is a valid news article text.") == True
    
    # Null text
    with pytest.raises(ValueError, match="cannot be Null/None"):
        validator.validate(None)
        
    # Non-string text
    with pytest.raises(ValueError, match="must be a string"):
        validator.validate(12345)
        
    # Empty text
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        validator.validate("   ")
        
    # Too short text
    with pytest.raises(ValueError, match="too short"):
        validator.validate("Short")
        
    # Too long text
    with pytest.raises(ValueError, match="too long"):
        validator.validate("A" * 105)

# 2. Test ConfidenceCalculator
class MockProbModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])

class MockMarginModel:
    def decision_function(self, X):
        return np.array([1.5])

class MockFallbackModel:
    pass

def test_confidence_calculator():
    calc = ConfidenceCalculator()
    
    # Probability model
    model_prob = MockProbModel()
    assert calc.calculate(model_prob, None, 1) == 0.8
    assert calc.calculate(model_prob, None, 0) == 0.2
    
    # Margin model
    model_margin = MockMarginModel()
    # 1.5 decision function maps to sigmoid(1.5) = 1.0 / (1.0 + exp(-1.5)) ≈ 0.81757
    conf = calc.calculate(model_margin, None, 1)
    assert 0.81 < conf < 0.82
    
    # Fallback model
    model_fallback = MockFallbackModel()
    assert calc.calculate(model_fallback, None, 1) == 1.0

# 3. Test ResponseBuilder
def test_response_builder():
    builder = ResponseBuilder()
    inference_metrics = {
        "latency_ms": 12.34,
        "throughput_sps": 81.0,
        "memory_used_mb": 0.05,
        "duration_sec": 0.01234
    }
    model_metadata = {"dataset_version": "1.0.1"}
    best_model_info = {"model_key": "svm"}
    
    resp = builder.build_response(
        prediction_label=1,
        confidence=0.85,
        inference_metrics=inference_metrics,
        model_metadata=model_metadata,
        best_model_info=best_model_info,
        evaluation_version="evaluation_v2"
    )
    
    assert resp["prediction"] == 1
    assert resp["label"] == "FAKE"
    assert resp["confidence"] == 0.85
    assert resp["model_name"] == "svm"
    assert resp["model_version"] == "1.0.1"
    assert resp["evaluation_version"] == "evaluation_v2"
    assert resp["prediction_time_ms"] == 12.34
    assert resp["throughput"] == 81.0
    assert resp["memory_usage"] == 0.05
    assert "timestamp" in resp

# 4. Test PredictionStatistics
def test_prediction_statistics():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_file = os.path.join(tmpdir, "stats.json")
        stats_mgr = PredictionStatistics(stats_file)
        
        # Test default load
        stats = stats_mgr.load_statistics()
        assert stats["total_predictions"] == 0
        
        # Add first prediction
        stats = stats_mgr.update_statistics(
            prediction_time_ms=10.0,
            memory_usage_mb=1.0,
            throughput_sps=100.0,
            confidence=0.8,
            model_used="svm"
        )
        assert stats["total_predictions"] == 1
        assert stats["average_prediction_time_ms"] == 10.0
        assert stats["average_memory_usage_mb"] == 1.0
        assert stats["average_throughput_sps"] == 100.0
        assert stats["average_confidence"] == 0.8
        
        # Add second prediction
        stats = stats_mgr.update_statistics(
            prediction_time_ms=20.0,
            memory_usage_mb=2.0,
            throughput_sps=50.0,
            confidence=0.9,
            model_used="svm"
        )
        assert stats["total_predictions"] == 2
        assert stats["average_prediction_time_ms"] == 15.0
        assert stats["average_memory_usage_mb"] == 1.5
        assert stats["average_throughput_sps"] == 75.0
        assert stats["average_confidence"] == 0.85

# 5. Test MetadataManager, HashGenerator, VersionManager, ReportGenerator, and Validator
def test_output_managers():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = os.path.join(tmpdir, "metadata.json")
        hash_file = os.path.join(tmpdir, "hashes.json")
        ver_file = os.path.join(tmpdir, "versions.json")
        report_file = os.path.join(tmpdir, "report.md")
        
        # Metadata Manager
        meta_mgr = MetadataManager(tmpdir)
        meta_mgr.save_metadata(
            file_path=meta_file,
            model_name="svm",
            training_version="train_v1",
            evaluation_version="eval_v1",
            feature_version="feat_v1",
            prediction_version="pred_v1"
        )
        assert os.path.exists(meta_file)
        
        # Hash Generator
        hash_gen = HashGenerator(hash_file)
        hashes = hash_gen.generate_hashes({"metadata": meta_file})
        assert "metadata" in hashes
        assert os.path.exists(hash_file)
        
        # Version Manager
        ver_mgr = VersionManager(ver_file)
        pred_ver = ver_mgr.register_version(
            training_version="train_v1",
            evaluation_version="eval_v1",
            model_used="svm",
            hashes=hashes
        )
        assert pred_ver == "prediction_v1"
        assert os.path.exists(ver_file)
        
        # Report Generator
        rep_gen = ReportGenerator(report_file)
        latest_response = {
            "prediction": 1,
            "label": "FAKE",
            "confidence": 0.8,
            "model_name": "svm",
            "model_version": "1.0.0",
            "evaluation_version": "eval_v1",
            "prediction_time_ms": 10.0,
            "throughput": 100.0,
            "memory_usage": 0.5,
            "timestamp": "2026-07-19T00:00:00"
        }
        rep_gen.generate_report(latest_response, {}, hashes, [])
        assert os.path.exists(report_file)
        
        # Validator
        validator = PredictionValidator()
        valid, errors = validator.validate_prediction_output(latest_response)
        assert valid == True
        assert len(errors) == 0
