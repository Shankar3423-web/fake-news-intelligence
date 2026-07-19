import os
import sys
import json
import logging
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.admin_review.admin_review_config import AdminReviewConfig
from ml.admin_review.review_validator import ReviewValidator
from ml.admin_review.admin_review_pipeline import run_admin_review_pipeline
from ml.admin_review.validator import AdminReviewOutputsValidator

def verify() -> bool:
    print("Starting Phase 11 Admin Review System Verification...")
    
    config_path = "ml/admin_review/admin_review_config.yaml"
    config = AdminReviewConfig(config_path)
    
    # 1. Verifying Configuration Loading
    print("1. Verifying configuration loading...")
    if "APPROVED" not in config.allowed_review_states:
        print("FAILED: Config allowed_review_states missing 'APPROVED'")
        return False
    if "REJECTED" not in config.allowed_review_states:
        print("FAILED: Config allowed_review_states missing 'REJECTED'")
        return False
    if "PENDING" not in config.allowed_review_states:
        print("FAILED: Config allowed_review_states missing 'PENDING'")
        return False
    print("Configuration loaded and verified successfully.")

    # 2. Verifying Input Validation Rules
    print("2. Verifying input validation rules...")
    validator = ReviewValidator(allowed_states=config.allowed_review_states)
    
    invalid_cases = [
        # Invalid status
        {
            "feedback_id": "fb_123",
            "prediction": 0,
            "verification": "VERIFIED",
            "final_decision": "REAL",
            "feedback_value": "Correct",
            "review_status": "MAYBE", # Invalid
            "reviewer": "Admin Bob",
            "timestamp": "2026-07-19T10:00:00"
        },
        # Empty reviewer
        {
            "feedback_id": "fb_123",
            "prediction": 0,
            "verification": "VERIFIED",
            "final_decision": "REAL",
            "feedback_value": "Correct",
            "review_status": "APPROVED",
            "reviewer": "   ", # Invalid
            "timestamp": "2026-07-19T10:00:00"
        },
        # Invalid timestamp format
        {
            "feedback_id": "fb_123",
            "prediction": 0,
            "verification": "VERIFIED",
            "final_decision": "REAL",
            "feedback_value": "Correct",
            "review_status": "APPROVED",
            "reviewer": "Admin Bob",
            "timestamp": "yesterday" # Invalid
        }
    ]
    for case in invalid_cases:
        ok, errs = validator.validate_review(case)
        if ok:
            print(f"FAILED: Validator accepted invalid case: {case}")
            return False
            
    valid_case = {
        "feedback_id": "fb_123",
        "prediction": 0,
        "verification": "VERIFIED",
        "final_decision": "REAL",
        "feedback_value": "Correct",
        "review_status": "APPROVED",
        "reviewer": "Admin Bob",
        "timestamp": "2026-07-19T10:00:00"
    }
    ok, errs = validator.validate_review(valid_case)
    if not ok:
        print(f"FAILED: Validator rejected valid case: {errs}")
        return False
    print("Input validation verified successfully.")

    # 3. Executing E2E Admin Review Pipeline
    print("3. Executing end-to-end admin review pipeline...")
    # Load feedback history to get actual IDs to review
    feedback_csv = config.get_path("feedback_history_csv")
    if not os.path.exists(feedback_csv) or os.path.getsize(feedback_csv) == 0:
        # Create a mock feedback history file if it doesn't exist
        print("WARNING: Feedback history file not found or empty. Generating mock feedback history...")
        os.makedirs(os.path.dirname(feedback_csv), exist_ok=True)
        mock_feedback = [
            {"Feedback ID": f"fb_{i}", "Timestamp": "2026-07-19T10:00:00", "Prediction": i % 2, "Verification": "VERIFIED", "Decision": "REAL" if i % 2 == 0 else "FAKE", "Feedback": "Correct", "Comment": "Mock comment"}
            for i in range(1, 8)
        ]
        df_mock = pd.DataFrame(mock_feedback)
        df_mock.to_csv(feedback_csv, index=False, encoding="utf-8")

    try:
        # Read the feedback IDs
        df_feed = pd.read_csv(feedback_csv)
        feedback_ids = df_feed["Feedback ID"].tolist()
        
        # We need at least 5 approvals to test retraining threshold
        # Let's review the first 5 records as APPROVED, 6th as REJECTED, 7th as PENDING
        responses = []
        reviewer_name = "Admin Validator"
        
        for idx, fid in enumerate(feedback_ids):
            if idx < 5:
                status = "APPROVED"
                notes = f"Approved feedback record #{idx + 1}"
            elif idx == 5:
                status = "REJECTED"
                notes = "Rejected feedback record: invalid user comments"
            else:
                status = "PENDING"
                notes = "Setting to pending for further investigation"
                
            res = run_admin_review_pipeline(
                feedback_id=fid,
                review_status=status,
                reviewer=reviewer_name,
                notes=notes,
                config_path=config_path
            )
            responses.append(res)

        print("Pipeline execution completed.")
        
        # Validate that the last response succeeded
        last_res = responses[-1]
        if last_res["status"] != "SUCCESS":
            print(f"FAILED: Pipeline returned status FAILED. Error: {last_res.get('error')}")
            return False
            
    except Exception as e:
        print(f"Pipeline execution FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Checking Output Directories & Files Existence
    print("4. Checking output files and registries...")
    output_validator = AdminReviewOutputsValidator(config)
    is_valid_outputs, output_errors = output_validator.validate_outputs()
    
    if not is_valid_outputs:
        print("FAILED: Output verification errors detected:")
        for err in output_errors:
            print(f"  - {err}")
        return False
    print("All output files and schemas verified successfully.")

    # 5. Checking Unit Test Readiness
    print("5. Checking unit test readiness...")
    if "PYTEST_CURRENT_TEST" in os.environ:
        print("Active pytest session detected. Skipping nested pytest execution.")
    else:
        try:
            import pytest
            # Run tests in the package programmatically
            test_path = os.path.join(os.path.dirname(__file__), "tests", "test_admin_review.py")
            print(f"Executing pytest tests on: {test_path}")
            exit_code = pytest.main(["-v", test_path])
            if exit_code != 0:
                print(f"FAILED: pytest unit tests failed with exit code {exit_code}")
                return False
            print("Unit test suite ran successfully.")
        except Exception as e:
            print(f"FAILED to run unit tests: {e}")
            return False

    print("\n====================================================")
    print("PHASE 11 ADMIN REVIEW VERIFICATION PASSED")
    print("====================================================")
    
    # Close any logging locks
    from ml.admin_review.admin_review_logger import shutdown_logger
    shutdown_logger()
    
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
