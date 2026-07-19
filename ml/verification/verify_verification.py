import os
import json
import sys
import pandas as pd
from datetime import datetime

# Add the project root to path if running this file directly to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.verification.verification_pipeline import run_verification_pipeline
from ml.verification.verification_config import VerificationConfig
from ml.verification.verification_validator import VerificationValidator

def verify() -> bool:
    print("Starting Phase 9 Live News Verification Engine Verification...")
    
    config_path = "ml/verification/verification_config.yaml"
    config = VerificationConfig(config_path)
    validator = VerificationValidator()

    # 1. Check Configuration Loading
    print("1. Verifying configuration loading...")
    if config.timeout != 10:
        print(f"FAILED: Config timeout is {config.timeout}, expected 10")
        return False
    if config.similarity_threshold != 0.4:
        print(f"FAILED: Config similarity_threshold is {config.similarity_threshold}, expected 0.4")
        return False
    print("Configuration loaded and verified successfully.")

    # 2. Check API Keys Presence
    print("2. Verifying API keys presence...")
    api_keys = {
        "NEWS_API_KEY": os.getenv("NEWS_API_KEY", ""),
        "GNEWS_API_KEY": os.getenv("GNEWS_API_KEY", ""),
        "NEWSDATA_API_KEY": os.getenv("NEWSDATA_API_KEY", "")
    }
    for k, v in api_keys.items():
        if not v:
            print(f"WARNING: Environment variable {k} is missing or empty.")
        else:
            print(f"Found API key for {k}: {v[:5]}...{v[-5:] if len(v) > 5 else ''}")

    # 3. Define Test Context & Run Pipeline
    print("3. Executing end-to-end verification pipeline...")
    test_prediction_response = {
        "prediction": 0,
        "label": "REAL",
        "confidence": 0.8944,
        "model_name": "svm",
        "model_version": "1.0.0",
        "evaluation_version": "evaluation_v1",
        "prediction_time_ms": 15.2,
        "throughput": 50000.0,
        "memory_usage": 0.05,
        "timestamp": datetime.now().isoformat()
    }
    
    test_article_text = (
        "The central government has officially declared a massive economic relief plan today. "
        "The newly announced initiatives offer tax subsidies for small business owners and massive infrastructure grants. "
        "Markets responded positively immediately after the announcement. "
        "Treasury officials confirmed the package will begin implementation early next month."
    )
    
    try:
        response = run_verification_pipeline(
            article_text=test_article_text,
            prediction_response=test_prediction_response,
            article_title="Government Economic Subsidies Package",
            config_path=config_path
        )
        print("Pipeline execution completed.")
    except Exception as e:
        print(f"Pipeline execution FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Check Response Structure & Values
    print("4. Validating pipeline response schema...")
    res_ok, res_errs = validator.validate_verification_response(response)
    if not res_ok:
        print(f"Verification response validation FAILED: {res_errs}")
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
        ("Cache Directory", config.get_path("cache_dir")),
        ("History Directory", config.get_path("history_dir")),
        ("Charts Directory", config.get_path("charts_dir")),
        
        ("History CSV", config.get_path("verification_history_csv")),
        ("Statistics JSON", config.get_path("verification_statistics_json")),
        ("Report MD", config.get_path("verification_report_md")),
        ("Metadata JSON", config.get_path("verification_metadata_json")),
        ("Hashes JSON", config.get_path("verification_hashes_json")),
        ("Versions JSON", config.get_path("verification_versions_json"))
    ]

    for label, p in paths_to_verify:
        if not os.path.exists(p):
            print(f"Verification FAILED: {label} not found at {p}")
            return False
        if os.path.isfile(p) and os.path.getsize(p) == 0:
            print(f"Verification FAILED: {label} file is empty at {p}")
            return False

    print("All output folders and files exist and are populated.")

    # 6. Verify History DB Structure
    print("6. Verifying verification history CSV format...")
    try:
        df = pd.read_csv(config.get_path("verification_history_csv"))
        expected_cols = [
            "Timestamp", "Input Hash", "Prediction", "Verification Status",
            "Evidence Score", "Similarity", "Providers"
        ]
        for col in expected_cols:
            if col not in df.columns:
                print(f"FAILED: History CSV missing column '{col}'")
                return False
    except Exception as e:
        print(f"FAILED to read/parse History CSV: {e}")
        return False
    print("Verification history database structured successfully.")

    # 7. Check Hashes Registry
    print("7. Verifying hashes registry file...")
    try:
        with open(config.get_path("verification_hashes_json"), "r", encoding="utf-8") as f:
            hashes = json.load(f)
        if not isinstance(hashes, dict) or len(hashes) == 0:
            print("FAILED: Hashes registry is invalid or empty.")
            return False
        print(f"Hashes registered: {list(hashes.keys())}")
    except Exception as e:
        print(f"FAILED to parse hashes registry: {e}")
        return False

    # 8. Check Versions Registry
    print("8. Verifying versions registry database...")
    try:
        with open(config.get_path("verification_versions_json"), "r", encoding="utf-8") as f:
            versions = json.load(f)
        if not isinstance(versions, list) or len(versions) == 0:
            print("FAILED: Versions registry is invalid or empty.")
            return False
        print(f"Versions registered: {[v['version'] for v in versions]}")
    except Exception as e:
        print(f"FAILED to parse versions registry: {e}")
        return False

    # 9. Verify Rendered Chart Files (conditional on matplotlib availability)
    print("9. Checking generated visualization charts...")
    try:
        import matplotlib
        matplotlib_installed = True
    except ImportError:
        matplotlib_installed = False
        print("WARNING: matplotlib is not installed. Skipping chart existence checks.")

    if matplotlib_installed:
        chart_files = [
            "provider_response_time.png",
            "similarity_distribution.png",
            "provider_distribution.png",
            "verification_results.png"
        ]
        for chart in chart_files:
            chart_path = os.path.join(config.get_path("charts_dir"), chart)
            if not os.path.exists(chart_path) or os.path.getsize(chart_path) == 0:
                print(f"FAILED: Chart {chart} missing or empty at {chart_path}")
                return False
        print("All required charts were rendered and verified successfully.")

    print("\n====================================================")
    print("PHASE 9 LIVE NEWS VERIFICATION VERIFICATION PASSED")
    print("====================================================")
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
