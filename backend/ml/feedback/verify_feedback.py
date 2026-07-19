import os
import json
import sys
import pandas as pd

# Add the project root to path if running this file directly to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.feedback.feedback_config import FeedbackConfig
from ml.feedback.feedback_validator import FeedbackValidator
from ml.feedback.feedback_pipeline import run_feedback_pipeline

def verify() -> bool:
    print("Starting Phase 10 Feedback Collection System Verification...")
    
    config_path = "ml/feedback/feedback_config.yaml"
    config = FeedbackConfig(config_path)
    validator = FeedbackValidator(
        allowed_values=config.allowed_feedback_values,
        min_comment_len=config.min_comment_length,
        max_comment_len=config.max_comment_length,
        allow_empty_comments=config.allow_empty_comments
    )

    # 1. Check Configuration Loading
    print("1. Verifying configuration loading...")
    if config.max_comment_length != 500:
        print(f"FAILED: Config max_comment_length is {config.max_comment_length}, expected 500")
        return False
    if config.min_comment_length != 3:
        print(f"FAILED: Config min_comment_length is {config.min_comment_length}, expected 3")
        return False
    if "Correct" not in config.allowed_feedback_values:
        print("FAILED: Config allowed_feedback_values missing 'Correct'")
        return False
    print("Configuration loaded and verified successfully.")

    # 2. Check Input Validation
    print("2. Verifying input validation rules...")
    invalid_cases = [
        # Invalid feedback value
        {"prediction": 1, "prediction_confidence": 0.85, "verification_status": "VERIFIED", "evidence_score": 0.9, "similarity_score": 0.8, "final_decision": "REAL", "user_feedback": "Maybe"},
        # Out of range confidence
        {"prediction": 1, "prediction_confidence": 1.5, "verification_status": "VERIFIED", "evidence_score": 0.9, "similarity_score": 0.8, "final_decision": "REAL", "user_feedback": "Correct"},
        # Comment too short
        {"prediction": 1, "prediction_confidence": 0.85, "verification_status": "VERIFIED", "evidence_score": 0.9, "similarity_score": 0.8, "final_decision": "REAL", "user_feedback": "Correct", "comment": "hi"}
    ]
    for case in invalid_cases:
        ok, errs = validator.validate_inputs(case)
        if ok:
            print(f"FAILED: Validator accepted invalid case: {case}")
            return False
            
    valid_case = {
        "prediction": 1, 
        "prediction_confidence": 0.85, 
        "verification_status": "VERIFIED", 
        "evidence_score": 0.9, 
        "similarity_score": 0.8, 
        "final_decision": "REAL", 
        "user_feedback": "Correct", 
        "comment": "Valid comment"
    }
    ok, errs = validator.validate_inputs(valid_case)
    if not ok:
        print(f"FAILED: Validator rejected valid case: {errs}")
        return False
    print("Input validation verified successfully.")

    # 3. Execute End-to-End Pipeline
    print("3. Executing end-to-end feedback collection pipeline...")
    try:
        response = run_feedback_pipeline(
            prediction=0,
            prediction_confidence=0.9234,
            verification_status="PARTIALLY VERIFIED",
            evidence_score=0.7123,
            similarity_score=0.6845,
            final_decision="SUSPECTED REAL",
            user_feedback="Correct",
            comment="The evidence matches real world statements.",
            config_path=config_path
        )
        print("Pipeline execution completed.")
    except Exception as e:
        print(f"Pipeline execution FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    if response["status"] != "SUCCESS":
        print(f"FAILED: Pipeline returned FAILED status. Error: {response.get('error')}")
        return False

    # 4. Check Response Structure & Values
    print("4. Validating pipeline response schema...")
    expected_response_keys = ["feedback_id", "status", "record", "warnings", "error", "version", "file_paths", "hashes"]
    for key in expected_response_keys:
        if key not in response:
            print(f"FAILED: Response missing key '{key}'")
            return False
    print("Pipeline response schema validated successfully.")

    # 5. Check Directory & Output Files Existence
    print("5. Checking output files and registries...")
    paths_to_verify = [
        ("Logs Directory", config.get_path("logs_dir")),
        ("Reports Directory", config.get_path("reports_dir")),
        ("Statistics Directory", config.get_path("statistics_dir")),
        ("Metadata Directory", config.get_path("metadata_dir")),
        ("Hashes Directory", config.get_path("hashes_dir")),
        ("Versions Directory", config.get_path("versions_dir")),
        ("History Directory", config.get_path("history_dir")),
        ("Charts Directory", config.get_path("charts_dir")),
        
        ("History CSV", config.get_path("feedback_history_csv")),
        ("Statistics JSON", config.get_path("feedback_statistics_json")),
        ("Report MD", config.get_path("feedback_report_md")),
        ("Metadata JSON", config.get_path("feedback_metadata_json")),
        ("Hashes JSON", config.get_path("feedback_hashes_json")),
        ("Versions JSON", config.get_path("feedback_versions_json"))
    ]

    for label, p in paths_to_verify:
        if not os.path.exists(p):
            print(f"Verification FAILED: {label} not found at {p}")
            return False
        if os.path.isfile(p) and os.path.getsize(p) == 0:
            print(f"Verification FAILED: {label} file is empty at {p}")
            return False

    print("All output folders and files exist and are populated.")

    # 6. Verify History CSV Structure
    print("6. Verifying feedback history CSV format...")
    try:
        df = pd.read_csv(config.get_path("feedback_history_csv"))
        expected_cols = [
            "Feedback ID", "Timestamp", "Prediction", "Verification",
            "Decision", "Feedback", "Comment"
        ]
        for col in expected_cols:
            if col not in df.columns:
                print(f"FAILED: History CSV missing column '{col}'")
                return False
    except Exception as e:
        print(f"FAILED to read/parse History CSV: {e}")
        return False
    print("Feedback history CSV structured successfully.")

    # 7. Check Hashes Registry
    print("7. Verifying hashes registry file...")
    try:
        with open(config.get_path("feedback_hashes_json"), "r", encoding="utf-8") as f:
            hashes = json.load(f)
        if not isinstance(hashes, dict) or len(hashes) == 0:
            print("FAILED: Hashes registry is invalid or empty.")
            return False
        required_hash_keys = ["history", "metadata", "statistics", "report", "versions"]
        for hk in required_hash_keys:
            if hk not in hashes:
                print(f"FAILED: Hashes missing key '{hk}'")
                return False
        print(f"Hashes registered: {list(hashes.keys())}")
    except Exception as e:
        print(f"FAILED to parse hashes registry: {e}")
        return False

    # 8. Check Versions Registry
    print("8. Verifying versions registry database...")
    try:
        with open(config.get_path("feedback_versions_json"), "r", encoding="utf-8") as f:
            versions = json.load(f)
        if not isinstance(versions, list) or len(versions) == 0:
            print("FAILED: Versions registry is invalid or empty.")
            return False
        print(f"Versions registered: {[v['version'] for v in versions]}")
    except Exception as e:
        print(f"FAILED to parse versions registry: {e}")
        return False

    # 9. Verify Rendered Chart Files
    print("9. Checking generated visualization charts...")
    try:
        import matplotlib
        matplotlib_installed = True
    except ImportError:
        matplotlib_installed = False
        print("WARNING: matplotlib is not installed. Skipping chart existence checks.")

    if matplotlib_installed:
        chart_files = [
            "feedback_distribution.png",
            "correct_vs_incorrect.png",
            "prediction_distribution.png",
            "verification_distribution.png",
            "decision_distribution.png"
        ]
        for chart in chart_files:
            chart_path = os.path.join(config.get_path("charts_dir"), chart)
            if not os.path.exists(chart_path) or os.path.getsize(chart_path) == 0:
                print(f"FAILED: Chart {chart} missing or empty at {chart_path}")
                return False
        print("All required charts were rendered and verified successfully.")

    # 10. Check Log Generation
    print("10. Checking log files...")
    log_file_path = config.get_path("feedback_pipeline_log")
    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        print(f"FAILED: Feedback pipeline log file is missing or empty at {log_file_path}")
        return False
    print("Log files generated successfully.")

    print("\n====================================================")
    print("PHASE 10 FEEDBACK SYSTEM VERIFICATION PASSED")
    print("====================================================")
    
    from ml.feedback.feedback_logger import shutdown_logger
    shutdown_logger()
    
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
