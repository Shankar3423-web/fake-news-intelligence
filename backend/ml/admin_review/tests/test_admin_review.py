import os
import json
import tempfile
import pytest
import pandas as pd
from datetime import datetime

from ml.admin_review.admin_review_config import AdminReviewConfig
from ml.admin_review.feedback_loader import FeedbackLoader
from ml.admin_review.review_validator import ReviewValidator
from ml.admin_review.history_manager import HistoryManager
from ml.admin_review.approved_dataset_manager import ApprovedDatasetManager
from ml.admin_review.approval_manager import ApprovalManager
from ml.admin_review.statistics_manager import StatisticsManager
from ml.admin_review.metadata_manager import MetadataManager
from ml.admin_review.report_generator import ReportGenerator
from ml.admin_review.hash_generator import HashGenerator
from ml.admin_review.version_manager import VersionManager
from ml.admin_review.visualization import AdminReviewVisualizer
from ml.admin_review.validator import AdminReviewOutputsValidator
from ml.admin_review.admin_review_pipeline import run_admin_review_pipeline
from ml.admin_review.verify_admin_review import verify

def test_review_validator():
    validator = ReviewValidator(allowed_states=["APPROVED", "REJECTED", "PENDING"])
    
    valid_review = {
        "feedback_id": "fb_123",
        "prediction": 0,
        "verification": "VERIFIED",
        "final_decision": "REAL",
        "feedback_value": "Correct",
        "review_status": "APPROVED",
        "reviewer": "Admin Alice",
        "timestamp": "2026-07-19T10:00:00"
    }
    
    is_ok, errs = validator.validate_review(valid_review)
    assert is_ok is True
    assert len(errs) == 0

    # Invalid status
    bad_status = valid_review.copy()
    bad_status["review_status"] = "INVALID_STATUS"
    is_ok, errs = validator.validate_review(bad_status)
    assert is_ok is False
    assert any("Invalid Review Status" in e for e in errs)

    # Empty reviewer name
    bad_reviewer = valid_review.copy()
    bad_reviewer["reviewer"] = "  "
    is_ok, errs = validator.validate_review(bad_reviewer)
    assert is_ok is False
    assert any("Reviewer Name" in e for e in errs)

    # Invalid timestamp
    bad_time = valid_review.copy()
    bad_time["timestamp"] = "2026-13-40T25:00:00"
    is_ok, errs = validator.validate_review(bad_time)
    assert is_ok is False

def test_approval_manager():
    mgr = ApprovalManager(min_retraining_records=3)
    
    # Not enough records
    assert mgr.is_eligible_for_retraining([{"Feedback ID": "fb_1"}]) is False
    
    # Enough records
    assert mgr.is_eligible_for_retraining([
        {"Feedback ID": "fb_1"},
        {"Feedback ID": "fb_2"},
        {"Feedback ID": "fb_3"}
    ]) is True

    # Check extract approved IDs
    history = [
        {"Feedback ID": "fb_1", "Review Status": "APPROVED"},
        {"Feedback ID": "fb_2", "Review Status": "REJECTED"},
        {"Feedback ID": "fb_3", "Review Status": "PENDING"},
        {"Feedback ID": "fb_1", "Review Status": "PENDING"}, # Updated to pending
        {"Feedback ID": "fb_3", "Review Status": "APPROVED"}  # Updated to approved
    ]
    approved_ids = mgr.extract_approved_ids(history)
    assert "fb_3" in approved_ids
    assert "fb_1" not in approved_ids
    assert "fb_2" not in approved_ids

def test_history_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "review_history.csv")
        mgr = HistoryManager(history_csv_path=csv_path)
        
        mgr.save_history("fb_1", "APPROVED", "Alice", "Looks real", "2026-07-19T10:00:00")
        mgr.save_history("fb_2", "REJECTED", "Bob", "Looks fake", "2026-07-19T10:01:00")
        
        history = mgr.load_history()
        assert len(history) == 2
        assert history[0]["Feedback ID"] == "fb_1"
        assert history[0]["Review Status"] == "APPROVED"
        assert history[0]["Reviewer"] == "Alice"
        assert history[1]["Review Notes"] == "Looks fake"

def test_approved_dataset_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "approved_feedback.csv")
        mgr = ApprovedDatasetManager(approved_csv_path=csv_path)
        
        feedback_records = [
            {"Feedback ID": "fb_1", "Timestamp": "2026-07-19", "Prediction": 1, "Verification": "VERIFIED", "Decision": "REAL", "Feedback": "Correct", "Comment": "Yes"},
            {"Feedback ID": "fb_2", "Timestamp": "2026-07-19", "Prediction": 0, "Verification": "VERIFIED", "Decision": "FAKE", "Feedback": "Correct", "Comment": "No"}
        ]
        
        # 1. Sync when fb_1 is approved
        mgr.sync_approved_dataset({"fb_1": "APPROVED", "fb_2": "REJECTED"}, feedback_records)
        approved = mgr.load_approved_dataset()
        assert len(approved) == 1
        assert approved[0]["Feedback ID"] == "fb_1"
        
        # 2. Sync when no records approved
        mgr.sync_approved_dataset({"fb_1": "REJECTED", "fb_2": "PENDING"}, feedback_records)
        approved = mgr.load_approved_dataset()
        assert len(approved) == 0

def test_statistics_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = os.path.join(tmpdir, "statistics.json")
        mgr = StatisticsManager(stats_file_path=stats_path)
        
        history = [
            {"Feedback ID": "fb_1", "Review Status": "APPROVED"},
            {"Feedback ID": "fb_2", "Review Status": "REJECTED"},
            {"Feedback ID": "fb_3", "Review Status": "PENDING"},
            {"Feedback ID": "fb_1", "Review Status": "REJECTED"} # Overwrites fb_1
        ]
        
        stats = mgr.calculate_and_save(history)
        assert stats["total_reviews"] == 3
        assert stats["approved_count"] == 0 # fb_1 is now REJECTED
        assert stats["rejected_count"] == 2 # fb_1 and fb_2
        assert stats["pending_count"] == 1  # fb_3
        assert stats["approval_rate"] == 0.0

def test_metadata_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = os.path.join(tmpdir, "metadata.json")
        mgr = MetadataManager(metadata_dir=tmpdir)
        
        metadata = mgr.save_metadata(meta_path, "1.0.0", "2.0.0", "1.0.0")
        assert os.path.exists(meta_path)
        assert metadata["pipeline_version"] == "1.0.0"
        assert metadata["system_version"] == "2.0.0"

def test_hash_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        hash_file = os.path.join(tmpdir, "hashes.json")
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Hello Admin")
            
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
        assert v1 == "review_v1"
        
        versions = mgr.load_versions()
        assert len(versions) == 1
        assert versions[0]["version"] == "review_v1"

def test_report_generator():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, "report.md")
        csv_path = os.path.join(tmpdir, "review_history.csv")
        
        history_mgr = HistoryManager(history_csv_path=csv_path)
        history_mgr.save_history("fb_1", "APPROVED", "Alice", "Good", "2026-07-19T10:00:00")
        
        report_gen = ReportGenerator(report_file_path=report_path, history_csv_path=csv_path)
        
        record = {
            "feedback_id": "fb_1",
            "review_status": "APPROVED",
            "reviewer": "Alice",
            "timestamp": "2026-07-19T10:00:00"
        }
        stats = {
            "Total Reviews": 1,
            "Approved Count": 1,
            "Rejected Count": 0,
            "Pending Count": 0,
            "Approval Rate": 1.0
        }
        hashes = {"history": "h1", "metadata": "h2"}
        
        report_gen.generate_report(record, stats, hashes, ["Test warning"])
        assert os.path.exists(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Phase 11 - Admin Review Report" in content
        assert "APPROVED" in content
        assert "Test warning" in content

def test_pipeline_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Convert path delimiters for YAML config string safety
        tmpdir = tmpdir.replace("\\", "/")
        
        # Setup dummy feedback history file
        feedback_csv = f"{tmpdir}/feedback_history.csv"
        df_feed = pd.DataFrame([{
            "Feedback ID": "fb_test",
            "Timestamp": "2026-07-19T10:00:00",
            "Prediction": 1,
            "Verification": "VERIFIED",
            "Decision": "REAL",
            "Feedback": "Correct",
            "Comment": "Perfect prediction!"
        }])
        df_feed.to_csv(feedback_csv, index=False, encoding="utf-8")

        config_yaml = f"""
admin_review:
  allowed_review_states:
    - "APPROVED"
    - "REJECTED"
    - "PENDING"
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
  approved_dir: "{tmpdir}/approved"
  hashes_dir: "{tmpdir}/hashes"
  versions_dir: "{tmpdir}/versions"
  charts_dir: "{tmpdir}/charts"
  
  feedback_history_csv: "{feedback_csv}"
  admin_review_history_csv: "{tmpdir}/history/admin_review_history.csv"
  approved_feedback_csv: "{tmpdir}/approved/approved_feedback.csv"
  admin_review_statistics_json: "{tmpdir}/statistics/admin_review_statistics.json"
  admin_review_metadata_json: "{tmpdir}/metadata/admin_review_metadata.json"
  admin_review_report_md: "{tmpdir}/reports/admin_review_report.md"
  admin_review_hashes_json: "{tmpdir}/hashes/admin_review_hashes.json"
  admin_review_versions_json: "{tmpdir}/versions/admin_review_versions.json"
  admin_review_pipeline_log: "{tmpdir}/logs/admin_review_pipeline.log"
"""
        config_path = os.path.join(tmpdir, "test_admin_config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_yaml)
            
        res = run_admin_review_pipeline(
            feedback_id="fb_test",
            review_status="APPROVED",
            reviewer="Alice",
            notes="Validating feedback context",
            config_path=config_path
        )
        
        assert res["status"] == "SUCCESS"
        assert res["version"] == "review_v1"
        assert os.path.exists(res["file_paths"]["history_csv"])
        assert os.path.exists(res["file_paths"]["approved_csv"])
        assert os.path.exists(res["file_paths"]["statistics_json"])
        assert os.path.exists(res["file_paths"]["metadata_json"])
        assert os.path.exists(res["file_paths"]["report_md"])
        assert os.path.exists(res["file_paths"]["hashes_json"])
        assert os.path.exists(res["file_paths"]["versions_json"])
        
        # Verify clean logging shutdown
        from ml.admin_review.admin_review_logger import shutdown_logger
        shutdown_logger()

def test_verify_script():
    # Calling verification logic
    res = verify()
    assert res is True
    
    from ml.admin_review.admin_review_logger import shutdown_logger
    shutdown_logger()
