import os
import sys
import json
import hashlib
import pandas as pd
from typing import Tuple, List

def verify_feature_selection_integrity(project_root: str = ".") -> Tuple[bool, List[str]]:
    """
    Runs production verification checks on all outputs of the feature selection pipeline.
    
    Returns:
        Tuple of (success: bool, list of failure reason strings)
    """
    failures = []
    
    # 1. Verify existence of required paths
    paths_to_check = {
        "input_dataset": "ml/feature_engineering/processed/feature_dataset_v1.csv",
        "output_dataset": "ml/feature_selection/processed/selected_feature_dataset_v1.csv",
        "selected_names": "ml/feature_selection/processed/selected_feature_names.json",
        "feature_rankings": "ml/feature_selection/processed/feature_rankings.json",
        "summary": "ml/feature_selection/processed/feature_selection_summary.json",
        "statistics": "ml/feature_selection/statistics/selection_statistics.json",
        "profile": "ml/feature_selection/statistics/selection_profile.json",
        "hash": "ml/feature_selection/statistics/selection_hash.json",
        "versions": "ml/feature_selection/statistics/selection_versions.json",
        "quality_report": "ml/feature_selection/reports/feature_selection_report.md"
    }

    for key, rel_path in paths_to_check.items():
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            failures.append(f"Required file does not exist: {rel_path}")

    # Check models
    selectors = ["variance", "correlation", "mutual_information", "chi_square", "random_forest", "rfe"]
    for selector in selectors:
        model_path = os.path.join(project_root, "ml/feature_selection/models", f"{selector}_selector.joblib")
        if not os.path.exists(model_path):
            failures.append(f"Selector model object not found: {model_path}")

    input_path = os.path.join(project_root, paths_to_check["input_dataset"])
    output_path = os.path.join(project_root, paths_to_check["output_dataset"])
    selected_names_path = os.path.join(project_root, paths_to_check["selected_names"])
    hash_meta_path = os.path.join(project_root, paths_to_check["hash"])
    version_meta_path = os.path.join(project_root, paths_to_check["versions"])
    
    if not os.path.exists(output_path):
        failures.append("Output selected dataset missing, skipping data-level integrity checks.")
        return False, failures

    # 2. Check input row count and sample matching
    input_row_count = 0
    df_in = None
    if os.path.exists(input_path):
        try:
            df_in = pd.read_csv(input_path, usecols=["id", "label"])
            input_row_count = len(df_in)
        except Exception as e:
            failures.append(f"Failed to read input feature engineering dataset: {e}")

    try:
        # Load output dataset
        df_out = pd.read_csv(output_path)
        output_row_count = len(df_out)
        
        # Verify row counts match
        if input_row_count > 0 and output_row_count != input_row_count:
            failures.append(f"Row count mismatch! Input has {input_row_count} rows, but output has {output_row_count} rows.")
        
        # Verify sample order (check id equality row-for-row)
        if df_in is not None and not df_in["id"].equals(df_out["id"]):
            failures.append("Sample order mismatch! Output 'id' column does not match input 'id' column row-for-row.")

        # Verify labels are preserved row-for-row
        if df_in is not None and not df_in["label"].equals(df_out["label"]):
            failures.append("Label mismatch! Output 'label' column does not match input 'label' column row-for-row.")

        # Required base columns exist
        required_cols = ["id", "label", "cleaned_text"]
        missing_cols = [col for col in required_cols if col not in df_out.columns]
        if missing_cols:
            failures.append(f"Output dataset is missing required base columns: {missing_cols}")

        # Check that expected selected features exist
        if os.path.exists(selected_names_path):
            with open(selected_names_path, "r", encoding="utf-8") as f:
                selected_features = json.load(f)
            
            missing_features = [col for col in selected_features if col not in df_out.columns]
            if missing_features:
                failures.append(f"Output dataset is missing expected selected features: {missing_features}")
                
            # Check for null/NaN values in feature columns
            feature_cols_present = [col for col in selected_features if col in df_out.columns]
            if feature_cols_present:
                null_counts = df_out[feature_cols_present].isnull().sum()
                cols_with_nulls = null_counts[null_counts > 0]
                if not cols_with_nulls.empty:
                    for col, count in cols_with_nulls.items():
                        failures.append(f"Feature column '{col}' contains {count} null/NaN values.")

        # Check label values
        invalid_labels = df_out[~df_out["label"].isin([0, 1])]["label"].unique().tolist()
        if invalid_labels:
            failures.append(f"Labels contain invalid values (only 0 and 1 allowed): {invalid_labels}")

    except Exception as e:
        failures.append(f"Error reading and verifying output selected dataset content: {str(e)}")

    # 3. Verify hash metadata file matches actual hashes
    if os.path.exists(hash_meta_path):
        try:
            with open(hash_meta_path, "r", encoding="utf-8") as f:
                hash_data = json.load(f)
            
            files_to_hash = {
                "selected_feature_dataset_v1.csv": output_path,
                "selected_feature_names.json": selected_names_path,
                "feature_rankings.json": os.path.join(project_root, paths_to_check["feature_rankings"]),
                "feature_selection_summary.json": os.path.join(project_root, paths_to_check["summary"]),
                "selection_statistics.json": os.path.join(project_root, paths_to_check["statistics"]),
                "selection_profile.json": os.path.join(project_root, paths_to_check["profile"]),
            }
            
            for key, path in files_to_hash.items():
                if os.path.exists(path):
                    sha256 = hashlib.sha256()
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256.update(chunk)
                    actual_digest = sha256.hexdigest()
                    
                    if hash_data.get(key) != actual_digest:
                        failures.append(f"Hash mismatch for {key}! Metadata has {hash_data.get(key)}, but actual hash is {actual_digest}")
        except Exception as e:
            failures.append(f"Failed to verify hash file integrity: {str(e)}")

    # 4. Verify version metadata exists
    if os.path.exists(version_meta_path):
        try:
            with open(version_meta_path, "r", encoding="utf-8") as f:
                versions = json.load(f)
            if not isinstance(versions, list) or len(versions) == 0:
                failures.append("Versions file is empty or formatted incorrectly.")
            else:
                last_version = versions[-1]
                if last_version.get("files", {}).get("selected_feature_dataset_v1.csv") is None:
                    failures.append("Last logged version does not reference the correct feature selection output hash.")
        except Exception as e:
            failures.append(f"Failed to verify version file integrity: {str(e)}")

    return len(failures) == 0, failures

if __name__ == "__main__":
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    passed, errors = verify_feature_selection_integrity(project_root=project_root_dir)
    
    if passed:
        print("\n" + "="*50)
        print("PHASE 5 FEATURE SELECTION VERIFICATION PASSED")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("PHASE 5 FEATURE SELECTION VERIFICATION FAILED")
        print("="*50)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*50 + "\n")
        sys.exit(1)
