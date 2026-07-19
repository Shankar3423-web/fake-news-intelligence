import os
import json
import tempfile
import pytest
import pandas as pd

from ml.feedback.feedback_config import FeedbackConfig
from ml.feedback.feedback_validator import FeedbackValidator
from ml.feedback.feedback_sanitizer import FeedbackSanitizer
from ml.feedback.feedback_manager import FeedbackManager
from ml.feedback.feedback_history import HistoryManager
from ml.feedback.feedback_statistics import StatisticsManager
from ml.feedback.feedback_metadata import MetadataManager
from ml.feedback.feedback_report import ReportGenerator
from ml.feedback.feedback_visualization import FeedbackVisualizer
from ml.feedback.feedback_hashes import HashGenerator
from ml.feedback.feedback_versions import VersionManager
from ml.feedback.response_builder import ResponseBuilder
from ml.feedback.feedback_pipeline import run_feedback_pipeline
from ml.feedback.verify_feedback import verify

def test_feedback_validator():
    validator = FeedbackValidator(
        allowed_values=["Correct", "Incorrect", "Unsure"],
        min_comment_len=3,
        max_comment_len=50,
        allow_empty_comments=True
    )
    
    # Valid input
    valid_input = {
        "prediction": 1,
        "prediction_confidence": 0.85,
        "verification_status": "VERIFIED",
        "evidence_score": 0.9,
        "similarity_score": 0.8,
        "final_decision": "REAL",
        "user_feedback": "Correct",
        "comment": "Nice job",
        "timestamp": "2026-07-19T10:00:00"
    }
    is_ok, errs = validator.validate_inputs(valid_input)
    assert is_ok is True
    assert len(errs) == 0

    # Invalid feedback value
    bad_feedback = valid_input.copy()
    bad_feedback["user_feedback"] = "Unknown"
    is_ok, errs = validator.validate_inputs(bad_feedback)
    assert is_ok is False
    assert any("Invalid Feedback Value" in e for e in errs)

    # Invalid confidence bounds
    bad_conf = valid_input.copy()
    bad_conf["prediction_confidence"] = 1.1
    is_ok, errs = validator.validate_inputs(bad_conf)
    assert is_ok is False

    # Comment too short
    bad_comment = valid_input.copy()
    bad_comment["comment"] = "a"
    is_ok, errs = validator.validate_inputs(bad_comment)
    assert is_ok is False
    assert any("Comment length" in e for e in errs)

    # Invalid timestamp format
    bad_time = valid_input.copy()
    bad_time["timestamp"] = "invalid-date-format"
    is_ok, errs = validator.validate_inputs(bad_time)
    assert is_ok is False
    assert any("valid ISO 8601 format" in e for e in errs)

def test_feedback_sanitizer():
    sanitizer = FeedbackSanitizer()
    
    # Unicode normalization and spaces
    dirty_comment = "  Some   dirty \u212c comment   "
    clean = sanitizer.sanitize_comment(dirty_comment)
    assert clean.startswith("Some dirty")
    assert clean.endswith("comment")
    assert "  " not in clean
    
    # HTML characters escaping
    html_comment = "<b>Bold</b> & <script>alert(1)</script>"
    clean_html = sanitizer.sanitize_comment(html_comment)
    assert "&lt;b&gt;" in clean_html
    assert "&amp;" in clean_html
    assert "<" not in clean_html

    # Control characters removal
    ctrl_comment = "Line\x00One\x1fTwo"
    clean_ctrl = sanitizer.sanitize_comment(ctrl_comment)
    assert "\x00" not in clean_ctrl
    assert "\x1f" not in clean_ctrl
    assert clean_ctrl == "LineOneTwo"

def test_feedback_manager():
    manager = FeedbackManager(system_version="1.2.3")
    
    record = manager.create_record(
        prediction=0,
        prediction_confidence=0.75,
        verification_status="PARTIALLY VERIFIED",
        evidence_score=0.5,
        similarity_score=0.6,
        final_decision="FAKE",
        user_feedback="Unsure",
        comment="Needs check",
        timestamp="2026-07-19T10:30:00"
    )
    
    assert record["feedback_id"].startswith("fb_")
    assert record["prediction"] == 0
    assert record["prediction_confidence"] == 0.75
    assert record["system_version"] == "1.2.3"
    assert record["comment"] == "Needs check"

def test_history_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "history.csv")
        mgr = HistoryManager(history_csv_path=csv_path)
        
        # Save first record (creates file and header)
        mgr.save_history("fb_1", "2026-07-19T10:00:00", 1, "VERIFIED", "REAL", "Correct", "First comment")
        assert os.path.exists(csv_path)
        
        # Save second record (appends)
        mgr.save_history("fb_2", "2026-07-19T10:01:00", 0, "NOT VERIFIED", "FAKE", "Incorrect", "Second comment")
        
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["Feedback ID", "Timestamp", "Prediction", "Verification", "Decision", "Feedback", "Comment"]
        assert df.iloc[0]["Feedback ID"] == "fb_1"
        assert df.iloc[1]["Feedback"] == "Incorrect"

def test_statistics_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = os.path.join(tmpdir, "statistics.json")
        mgr = StatisticsManager(stats_file_path=stats_path)
        
        # Test default
        defaults = mgr.load_statistics()
        assert defaults["total_feedback"] == 0
        
        # First update
        stats = mgr.update_statistics("Correct", 0.9, 0.8, 0.7)
        assert stats["total_feedback"] == 1
        assert stats["correct_count"] == 1
        assert stats["feedback_distribution"]["Correct"] == 1.0
        assert stats["average_prediction_confidence"] == 0.9
        assert stats["average_similarity"] == 0.8
        assert stats["average_evidence_score"] == 0.7
        
        # Second update
        stats = mgr.update_statistics("Incorrect", 0.8, 0.6, 0.5)
        assert stats["total_feedback"] == 2
        assert stats["correct_count"] == 1
        assert stats["incorrect_count"] == 1
        assert stats["feedback_distribution"]["Correct"] == 0.5
        assert stats["feedback_distribution"]["Incorrect"] == 0.5
        assert stats["average_prediction_confidence"] == 0.85
        assert stats["average_similarity"] == 0.7
        assert stats["average_evidence_score"] == 0.6

def test_metadata_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "metadata.json")
        mgr = MetadataManager(metadata_dir=tmpdir)
        
        metadata = mgr.save_metadata(meta_path, "1.0.0", "2.0.0", "1.0.0")
        assert os.path.exists(meta_path)
        assert metadata["pipeline_version"] == "1.0.0"
        assert metadata["feedback_version"] == "2.0.0"

def test_report_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.md")
        csv_path = os.path.join(tmpdir, "history.csv")
        
        # Setup mock history
        history_mgr = HistoryManager(history_csv_path=csv_path)
        history_mgr.save_history("fb_1", "2026-07-19T10:00:00", 1, "VERIFIED", "REAL", "Correct", "Excellent")
        
        report_gen = ReportGenerator(report_file_path=report_path, history_csv_path=csv_path)
        
        record = {
            "feedback_id": "fb_1",
            "timestamp": "2026-07-19T10:00:00",
            "prediction": 1,
            "prediction_confidence": 0.95,
            "verification_status": "VERIFIED",
            "evidence_score": 0.9,
            "similarity_score": 0.8,
            "final_decision": "REAL",
            "feedback_value": "Correct",
            "comment": "Excellent",
            "system_version": "1.0.0"
        }
        stats = {
            "total_feedback": 1,
            "correct_count": 1,
            "incorrect_count": 0,
            "unsure_count": 0,
            "average_prediction_confidence": 0.95,
            "average_similarity": 0.8,
            "average_evidence_score": 0.9,
            "feedback_distribution": {"Correct": 1.0}
        }
        hashes = {"history": "hash123", "metadata": "hash456"}
        
        report_gen.generate_report(record, stats, hashes, ["Test warning"])
        
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Phase 10 - Feedback Collection Report" in content
        assert "fb_1" in content
        assert "Excellent" in content
        assert "Test warning" in content

def test_visualization():
    with tempfile.TemporaryDirectory() as tmpdir:
        charts_dir = os.path.join(tmpdir, "charts")
        csv_path = os.path.join(tmpdir, "history.csv")
        
        # Populate history
        history_mgr = HistoryManager(history_csv_path=csv_path)
        history_mgr.save_history("fb_1", "2026-07-19T10:00:00", 1, "VERIFIED", "REAL", "Correct", "Awesome")
        
        vis = FeedbackVisualizer(charts_dir=charts_dir, history_csv_path=csv_path)
        stats = {
            "correct_count": 1,
            "incorrect_count": 0,
            "unsure_count": 0
        }
        
        vis.generate_charts(stats)
        
        try:
            import matplotlib
            matplotlib_installed = True
        except ImportError:
            matplotlib_installed = False
            
        if matplotlib_installed:
            assert os.path.exists(os.path.join(charts_dir, "feedback_distribution.png"))
            assert os.path.exists(os.path.join(charts_dir, "correct_vs_incorrect.png"))
            assert os.path.exists(os.path.join(charts_dir, "prediction_distribution.png"))
            assert os.path.exists(os.path.join(charts_dir, "verification_distribution.png"))
            assert os.path.exists(os.path.join(charts_dir, "decision_distribution.png"))

def test_hash_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        hash_file = os.path.join(tmpdir, "hashes.json")
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Hello Hash")
            
        gen = HashGenerator(hash_file_path=hash_file)
        hashes = gen.generate_hashes({"test_key": test_file})
        
        assert os.path.exists(hash_file)
        assert "test_key" in hashes
        assert len(hashes["test_key"]) == 64

def test_version_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        versions_file = os.path.join(tmpdir, "versions.json")
        mgr = VersionManager(versions_file_path=versions_file)
        
        assert len(mgr.load_versions()) == 0
        
        hashes = {"history": "hash1", "statistics": "hash2"}
        v1 = mgr.register_version(hashes)
        assert v1 == "feedback_v1"
        
        versions = mgr.load_versions()
        assert len(versions) == 1
        assert versions[0]["version"] == "feedback_v1"
        assert versions[0]["hashes"] == hashes

def test_response_builder():
    builder = ResponseBuilder()
    record = {"feedback_id": "fb_123", "feedback_value": "Correct"}
    res = builder.build_response(
        status="SUCCESS",
        record=record,
        warnings=["Warn1"],
        version="feedback_v1",
        file_paths={"history": "path1"},
        hashes={"history": "hash1"}
    )
    
    assert res["feedback_id"] == "fb_123"
    assert res["status"] == "SUCCESS"
    assert res["record"] == record
    assert "Warn1" in res["warnings"]
    assert res["version"] == "feedback_v1"
    assert res["file_paths"] == {"history": "path1"}
    assert res["hashes"] == {"history": "hash1"}

def test_pipeline_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = tmpdir.replace("\\", "/")
        config_yaml = f"""
feedback:
  max_comment_length: 50
  min_comment_length: 3
  allowed_feedback_values:
    - "Correct"
    - "Incorrect"
    - "Unsure"
  allow_empty_comments: true
  system_version: "1.0.0"

exports:
  enable_charts: true
  enable_reports: true
  enable_metadata: true
  enable_statistics: true
  enable_versioning: true
  enable_hash_generation: true

paths:
  output_dir: "{tmpdir}"
  logs_dir: "{tmpdir}/logs"
  reports_dir: "{tmpdir}/reports"
  statistics_dir: "{tmpdir}/statistics"
  metadata_dir: "{tmpdir}/metadata"
  history_dir: "{tmpdir}/history"
  hashes_dir: "{tmpdir}/hashes"
  versions_dir: "{tmpdir}/versions"
  charts_dir: "{tmpdir}/charts"
  
  feedback_history_csv: "{tmpdir}/history/feedback_history.csv"
  feedback_statistics_json: "{tmpdir}/statistics/feedback_statistics.json"
  feedback_metadata_json: "{tmpdir}/metadata/feedback_metadata.json"
  feedback_report_md: "{tmpdir}/reports/feedback_report.md"
  feedback_hashes_json: "{tmpdir}/hashes/feedback_hashes.json"
  feedback_versions_json: "{tmpdir}/versions/feedback_versions.json"
  feedback_pipeline_log: "{tmpdir}/logs/feedback_pipeline.log"
"""
        config_path = os.path.join(tmpdir, "test_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_yaml)
            
        res = run_feedback_pipeline(
            prediction=0,
            prediction_confidence=0.88,
            verification_status="VERIFIED",
            evidence_score=0.95,
            similarity_score=0.87,
            final_decision="REAL",
            user_feedback="Correct",
            comment="Awesome system response!",
            config_path=config_path
        )
        
        assert res["status"] == "SUCCESS"
        assert res["version"] == "feedback_v1"
        assert os.path.exists(res["file_paths"]["history_csv"])
        assert os.path.exists(res["file_paths"]["statistics_json"])
        assert os.path.exists(res["file_paths"]["metadata_json"])
        assert os.path.exists(res["file_paths"]["report_md"])
        assert os.path.exists(res["file_paths"]["hashes_json"])
        assert os.path.exists(res["file_paths"]["versions_json"])
        
        from ml.feedback.feedback_logger import shutdown_logger
        shutdown_logger()

def test_verify_script():
    # Verify execution of verification function returns True
    res = verify()
    assert res is True
    
    from ml.feedback.feedback_logger import shutdown_logger
    shutdown_logger()
