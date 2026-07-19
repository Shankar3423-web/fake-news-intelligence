import os
import sys
import json
import hashlib
import pandas as pd
from typing import Tuple, List

def verify_preprocessing_integrity(project_root: str = ".") -> Tuple[bool, List[str]]:
    """
    Runs production verification checks on all outputs of the NLP preprocessing pipeline.
    
    Returns:
        Tuple of (success: bool, list of failure reason strings)
    """
    failures = []
    
    # 1. Verify existence of required paths
    paths_to_check = {
        "input_dataset": "ml/dataset/processed/master_dataset_v1.csv",
        "output_dataset": "ml/preprocessing/processed/preprocessed_dataset_v1.csv",
        "statistics": "ml/preprocessing/statistics/preprocessing_statistics.json",
        "profile": "ml/preprocessing/statistics/preprocessing_profile.json",
        "hash": "ml/preprocessing/statistics/preprocessing_hash.json",
        "versions": "ml/preprocessing/statistics/preprocessing_versions.json",
        "quality_report": "ml/preprocessing/reports/preprocessing_quality_report.md"
    }

    for key, rel_path in paths_to_check.items():
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            failures.append(f"Required file does not exist: {rel_path}")

    # If output dataset doesn't exist, we cannot run data checks
    output_path = os.path.join(project_root, paths_to_check["output_dataset"])
    input_path = os.path.join(project_root, paths_to_check["input_dataset"])
    
    if not os.path.exists(output_path):
        failures.append("Output dataset missing, skipping data-level integrity checks.")
        return False, failures

    try:
        # Load output dataset
        df_out = pd.read_csv(output_path)
        
        # 2. Required columns exist
        required_cols = [
            "id", "title", "text", "cleaned_text", "label", "source", "category", 
            "author", "published_date", "language", "url", "dataset_origin"
        ]
        missing_cols = [col for col in required_cols if col not in df_out.columns]
        if missing_cols:
            failures.append(f"Output dataset is missing required columns: {missing_cols}")

        if not missing_cols:
            # 3. cleaned_text exists and is never empty/null
            empty_cleaned = (df_out["cleaned_text"].isnull() | (df_out["cleaned_text"].astype(str).str.strip() == "")).sum()
            if empty_cleaned > 0:
                failures.append(f"Found {empty_cleaned} records with empty/blank cleaned_text.")

            # 4. Labels only contain 0 and 1
            invalid_labels = df_out[~df_out["label"].isin([0, 1])]["label"].unique().tolist()
            if invalid_labels:
                failures.append(f"Labels contain invalid values (only 0 and 1 allowed): {invalid_labels}")

            # 5. Verify label values are unchanged for matching records
            if os.path.exists(input_path):
                df_in = pd.read_csv(input_path, usecols=["id", "label"])
                # Create a map of id -> label for quick checking
                in_label_map = dict(zip(df_in["id"], df_in["label"]))
                
                mismatched_labels = 0
                missing_in_input = 0
                for idx, row in df_out.iterrows():
                    row_id = row["id"]
                    if row_id not in in_label_map:
                        missing_in_input += 1
                    elif row["label"] != in_label_map[row_id]:
                        mismatched_labels += 1
                
                if missing_in_input > 0:
                    failures.append(f"Found {missing_in_input} record IDs in output that did not exist in the input dataset.")
                if mismatched_labels > 0:
                    failures.append(f"Found {mismatched_labels} records where labels were mutated/swapped during preprocessing.")

                # 6. Row count preserved check (accounting for rejected/failed rows)
                stats_path = os.path.join(project_root, paths_to_check["statistics"])
                if os.path.exists(stats_path):
                    with open(stats_path, "r", encoding="utf-8") as f:
                        stats_data = json.load(f)
                    
                    rows_processed = stats_data.get("Rows Processed", 0)
                    if len(df_out) != rows_processed:
                        failures.append(f"Output dataset row count ({len(df_out)}) does not match 'Rows Processed' in statistics ({rows_processed}).")

    except Exception as e:
        failures.append(f"Error reading and verifying output dataset content: {str(e)}")

    # 7. Verify hash metadata file content matches actual hash
    hash_meta_path = os.path.join(project_root, paths_to_check["hash"])
    if os.path.exists(hash_meta_path) and os.path.exists(output_path):
        try:
            with open(hash_meta_path, "r", encoding="utf-8") as f:
                hash_data = json.load(f)
            
            # Compute actual hash of output dataset
            sha256 = hashlib.sha256()
            with open(output_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            actual_digest = sha256.hexdigest()
            
            if hash_data.get("SHA256") != actual_digest:
                failures.append(f"Output dataset hash mismatch! Metadata file has {hash_data.get('SHA256')}, but actual dataset hash is {actual_digest}")
        except Exception as e:
            failures.append(f"Failed to verify hash file integrity: {str(e)}")

    # 8. Verify version metadata exists
    version_meta_path = os.path.join(project_root, paths_to_check["versions"])
    if os.path.exists(version_meta_path):
        try:
            with open(version_meta_path, "r", encoding="utf-8") as f:
                versions = json.load(f)
            if not isinstance(versions, list) or len(versions) == 0:
                failures.append("Versions file is empty or formatted incorrectly.")
            else:
                last_version = versions[-1]
                if last_version.get("Output Dataset") != paths_to_check["output_dataset"]:
                    failures.append("Last logged version does not reference the correct preprocessed output path.")
        except Exception as e:
            failures.append(f"Failed to verify version file integrity: {str(e)}")

    return len(failures) == 0, failures

if __name__ == "__main__":
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    passed, errors = verify_preprocessing_integrity(project_root=project_root_dir)
    
    if passed:
        print("\n" + "="*50)
        print("PHASE 3 PREPROCESSING VERIFICATION PASSED")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("PHASE 3 PREPROCESSING VERIFICATION FAILED")
        print("="*50)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*50 + "\n")
        sys.exit(1)
