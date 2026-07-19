import os
import json
import sys
import pandas as pd

# Add the project root to path if running this file directly to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.prediction.prediction_pipeline import run_prediction_pipeline
from ml.prediction.prediction_config import PredictionConfig

def verify() -> bool:
    print("Starting Phase 8 Prediction Engine Verification...")
    
    test_text = "Government announces new economic reforms to reduce inflation."
    config_path = "ml/prediction/prediction_config.yaml"
    config = PredictionConfig(config_path)
    
    # 1. Run Pipeline
    try:
        response = run_prediction_pipeline(test_text, config_path)
        print("Pipeline execution completed.")
    except Exception as e:
        print(f"Pipeline execution FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Check Response Structure
    expected_response_keys = [
        "prediction", "label", "confidence", "model_name", 
        "model_version", "evaluation_version", "prediction_time_ms", 
        "throughput", "memory_usage", "timestamp"
    ]
    for k in expected_response_keys:
        if k not in response:
            print(f"Verification FAILED: response missing key '{k}'")
            return False
            
    print("Response structure verified.")

    # 3. Verify Folder Structure & Outputs
    paths_to_verify = [
        ("Logs Directory", config.get_path("logs_dir")),
        ("Reports Directory", config.get_path("reports_dir")),
        ("Statistics Directory", config.get_path("statistics_dir")),
        ("Metadata Directory", config.get_path("metadata_dir")),
        ("Hashes Directory", config.get_path("hashes_dir")),
        ("Versions Directory", config.get_path("versions_dir")),
        ("Predictions Directory", config.get_path("predictions_dir")),
        
        ("Prediction History CSV", config.get_path("prediction_history_csv")),
        ("Prediction Statistics JSON", config.get_path("prediction_statistics_json")),
        ("Prediction Report MD", config.get_path("prediction_report_md")),
        ("Prediction Metadata JSON", config.get_path("prediction_metadata_json")),
        ("Prediction Hashes JSON", config.get_path("prediction_hashes_json")),
        ("Prediction Versions JSON", config.get_path("prediction_versions_json"))
    ]
    
    for label, p in paths_to_verify:
        if not os.path.exists(p):
            print(f"Verification FAILED: {label} not found at {p}")
            return False
        if os.path.isfile(p) and os.path.getsize(p) == 0:
            print(f"Verification FAILED: {label} file is empty at {p}")
            return False
            
    print("All output folders and files exist and are populated.")

    # 4. Check Prediction Export
    try:
        df = pd.read_csv(config.get_path("prediction_history_csv"))
        expected_cols = ["Timestamp", "Input Hash", "Prediction", "Confidence", "Model", "Prediction Time", "Latency", "Version"]
        for col in expected_cols:
            if col not in df.columns:
                print(f"Verification FAILED: prediction history missing column '{col}'")
                return False
    except Exception as e:
        print(f"Verification FAILED: could not parse prediction history: {e}")
        return False
        
    print("Prediction History structure verified.")

    # 5. Check Hashes Registry
    try:
        with open(config.get_path("prediction_hashes_json"), "r", encoding="utf-8") as f:
            hashes = json.load(f)
        if not isinstance(hashes, dict) or len(hashes) == 0:
            print("Verification FAILED: Hashes file is invalid or empty.")
            return False
    except Exception as e:
        print(f"Verification FAILED: could not parse hashes file: {e}")
        return False
        
    print("Hashes registry verified.")

    # 6. Check Versions Database
    try:
        with open(config.get_path("prediction_versions_json"), "r", encoding="utf-8") as f:
            versions = json.load(f)
        if not isinstance(versions, list) or len(versions) == 0:
            print("Verification FAILED: Versions file is invalid or empty.")
            return False
    except Exception as e:
        print(f"Verification FAILED: could not parse versions file: {e}")
        return False
        
    print("Versions registry verified.")

    print("\n====================================================")
    print("PHASE 8 PREDICTION ENGINE VERIFICATION PASSED")
    print("====================================================")
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
