import os
import sys
import re
import json
import hashlib
import pandas as pd
from typing import Tuple, List

def verify_pipeline_integrity(project_root: str = ".") -> Tuple[bool, List[str]]:
    """
    Runs production verification checks on all outputs of the dataset pipeline.
    
    Returns:
        Tuple of (success: bool, list of failure reason strings)
    """
    failures = []
    
    # 1. Verify existence of required paths
    paths_to_check = {
        "raw_isot_fake": "ml/dataset/raw/isot/Fake.csv",
        "raw_isot_true": "ml/dataset/raw/isot/True.csv",
        "raw_india_fake": "ml/dataset/raw/india/Indian_Fake_News_dataset.xlsx",
        "raw_india_true": "ml/dataset/raw/india/Indian_True_News_dataset.xlsx",
        
        "cleaned_isot_true": "ml/dataset/cleaned/isot_true_cleaned.csv",
        "cleaned_isot_fake": "ml/dataset/cleaned/isot_fake_cleaned.csv",
        "cleaned_india_true": "ml/dataset/cleaned/india_true_cleaned.csv",
        "cleaned_india_fake": "ml/dataset/cleaned/india_fake_cleaned.csv",
        
        "merged": "ml/dataset/merged/merged_dataset.csv",
        "master": "ml/dataset/processed/master_dataset_v1.csv",
        
        "statistics": "ml/dataset/statistics/dataset_statistics.json",
        "profile": "ml/dataset/statistics/dataset_profile.json",
        "hash": "ml/dataset/statistics/dataset_hash.json",
        "versions": "ml/dataset/versions/dataset_versions.json"
    }

    for key, rel_path in paths_to_check.items():
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            failures.append(f"File/Path does not exist: {rel_path}")

    # If master dataset doesn't exist, we can't do data integrity checks
    master_path = os.path.join(project_root, paths_to_check["master"])
    if not os.path.exists(master_path):
        failures.append("Master dataset file missing, skipping data-level integrity checks.")
        return False, failures

    try:
        # Load master dataset
        df = pd.read_csv(master_path)
        
        # 2. Required columns exist
        required_cols = [
            "id", "title", "text", "label", "source", "category", 
            "author", "published_date", "language", "url", "dataset_origin"
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            failures.append(f"Master dataset is missing required columns: {missing_cols}")

        if not missing_cols:
            # 3. Labels only contain 0 and 1
            invalid_labels = df[~df["label"].isin([0, 1])]["label"].unique().tolist()
            if invalid_labels:
                failures.append(f"Labels contain invalid values (only 0 and 1 allowed): {invalid_labels}")

            # 4. No duplicate IDs
            duplicate_ids_count = df["id"].duplicated().sum()
            if duplicate_ids_count > 0:
                failures.append(f"Found {duplicate_ids_count} duplicate IDs in the master dataset.")

            # 5. No empty titles
            empty_titles_count = (df["title"].isnull() | (df["title"].astype(str).str.strip() == "")).sum()
            if empty_titles_count > 0:
                failures.append(f"Found {empty_titles_count} records with empty/blank titles.")

            # 6. No empty texts
            empty_texts_count = (df["text"].isnull() | (df["text"].astype(str).str.strip() == "")).sum()
            if empty_texts_count > 0:
                failures.append(f"Found {empty_texts_count} records with empty/blank article text.")

            # 7. Dataset origin contains only ISOT or INDIA
            invalid_origins = df[~df["dataset_origin"].isin(["ISOT", "INDIA"])]["dataset_origin"].unique().tolist()
            if invalid_origins:
                failures.append(f"Found invalid dataset_origin values: {invalid_origins}")

            # 8. Dates are valid format (YYYY-MM-DD or fully empty/NaN is fine)
            date_series = df["published_date"].dropna().astype(str)
            if not date_series.empty:
                bad_dates_mask = ~date_series.str.match(r"^\d{4}-\d{2}-\d{2}$")
                bad_dates_count = bad_dates_mask.sum()
                if bad_dates_count > 0:
                    sample_bad_dates = date_series[bad_dates_mask].head(5).tolist()
                    failures.append(f"Found {bad_dates_count} records with invalid date format (YYYY-MM-DD required). Samples: {sample_bad_dates}")
                    
    except Exception as e:
        failures.append(f"Error reading and verifying master dataset content: {str(e)}")

    # 9. Verify hash metadata file content matches actual hash
    hash_meta_path = os.path.join(project_root, paths_to_check["hash"])
    if os.path.exists(hash_meta_path) and os.path.exists(master_path):
        try:
            with open(hash_meta_path, "r", encoding="utf-8") as f:
                hash_data = json.load(f)
            
            # Compute actual hash
            sha256 = hashlib.sha256()
            with open(master_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            actual_digest = sha256.hexdigest()
            
            if hash_data.get("SHA256") != actual_digest:
                failures.append(f"Dataset hash mismatch! Metadata file has {hash_data.get('SHA256')}, but actual master dataset hash is {actual_digest}")
        except Exception as e:
            failures.append(f"Failed to verify hash file integrity: {str(e)}")

    return len(failures) == 0, failures

if __name__ == "__main__":
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    passed, errors = verify_pipeline_integrity(project_root=project_root_dir)
    
    if passed:
        print("\n" + "="*50)
        print("PHASE 2 VERIFICATION PASSED")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("PHASE 2 VERIFICATION FAILED")
        print("="*50)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*50 + "\n")
        sys.exit(1)
