import os
import sys
import json
import hashlib
import pandas as pd
import joblib
import scipy.sparse
from typing import Tuple, List

def verify_feature_integrity(project_root: str = ".") -> Tuple[bool, List[str]]:
    """
    Runs production verification checks on all outputs of the feature engineering pipeline.
    
    Returns:
        Tuple of (success: bool, list of failure reason strings)
    """
    failures = []
    
    # 1. Verify existence of required paths
    paths_to_check = {
        "input_dataset": "ml/preprocessing/processed/preprocessed_dataset_v1.csv",
        "output_dataset": "ml/feature_engineering/processed/feature_dataset_v1.csv",
        "tfidf_vectorizer": "ml/feature_engineering/processed/tfidf_vectorizer.joblib",
        "tfidf_matrix": "ml/feature_engineering/processed/tfidf_matrix.joblib",
        "statistics": "ml/feature_engineering/statistics/feature_statistics.json",
        "profile": "ml/feature_engineering/statistics/feature_profile.json",
        "hash": "ml/feature_engineering/statistics/feature_hash.json",
        "versions": "ml/feature_engineering/statistics/feature_versions.json",
        "quality_report": "ml/feature_engineering/reports/feature_quality_report.md"
    }

    for key, rel_path in paths_to_check.items():
        abs_path = os.path.join(project_root, rel_path)
        if not os.path.exists(abs_path):
            failures.append(f"Required file does not exist: {rel_path}")

    input_path = os.path.join(project_root, paths_to_check["input_dataset"])
    output_path = os.path.join(project_root, paths_to_check["output_dataset"])
    tfidf_matrix_path = os.path.join(project_root, paths_to_check["tfidf_matrix"])
    tfidf_vec_path = os.path.join(project_root, paths_to_check["tfidf_vectorizer"])
    
    if not os.path.exists(output_path):
        failures.append("Output dataset missing, skipping data-level integrity checks.")
        return False, failures

    # 2. Check preprocessed input row count
    input_row_count = 0
    if os.path.exists(input_path):
        try:
            df_in = pd.read_csv(input_path, usecols=["id"])
            input_row_count = len(df_in)
        except Exception as e:
            failures.append(f"Failed to read input preprocessed dataset: {e}")

    try:
        # Load output dataset
        df_out = pd.read_csv(output_path)
        output_row_count = len(df_out)
        
        # Verify row counts match
        if input_row_count > 0 and output_row_count != input_row_count:
            failures.append(f"Row count mismatch! Input has {input_row_count} rows, but output has {output_row_count} rows.")

        # Required base columns exist
        required_cols = ["id", "label", "cleaned_text"]
        missing_cols = [col for col in required_cols if col not in df_out.columns]
        if missing_cols:
            failures.append(f"Output dataset is missing required base columns: {missing_cols}")

        # Check for engineered features presence
        expected_features = [
            "stat_word_count", "stat_char_count", "stat_sentence_count",
            "stat_avg_word_length", "stat_avg_sentence_length", "stat_vocabulary_size",
            "read_flesch_reading_ease", "read_flesch_kincaid_grade",
            "read_smog", "read_gunning_fog", "read_coleman_liau",
            "lex_diversity", "lex_unique_words", "lex_stopword_ratio",
            "lex_long_word_ratio", "lex_short_word_ratio",
            "ling_entity_count", "ling_noun_count", "ling_verb_count", "ling_adj_count",
            "ling_pos_noun_ratio", "ling_pos_verb_ratio", "ling_pos_adj_ratio",
            "ling_pos_adv_ratio", "ling_pos_pron_ratio",
            "sym_digit_count", "sym_uppercase_count", 
            "sym_punctuation_count", "sym_special_char_count"
        ]
        missing_features = [col for col in expected_features if col not in df_out.columns]
        if missing_features:
            failures.append(f"Output dataset is missing expected engineered features: {missing_features}")

        # Check for null/NaN values in feature columns
        feature_cols_present = [col for col in expected_features if col in df_out.columns]
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
        failures.append(f"Error reading and verifying output dataset content: {str(e)}")

    # 3. Check TF-IDF Sparse Matrix Integrity
    if os.path.exists(tfidf_matrix_path):
        try:
            matrix = joblib.load(tfidf_matrix_path)
            
            # Verify type is scipy sparse matrix
            if not scipy.sparse.issparse(matrix):
                failures.append(f"TF-IDF matrix at {paths_to_check['tfidf_matrix']} is NOT sparse! Type found: {type(matrix)}")
                
            # Verify rows count matches output dataset
            if scipy.sparse.issparse(matrix):
                matrix_rows = matrix.shape[0]
                if matrix_rows != output_row_count:
                    failures.append(f"TF-IDF matrix rows ({matrix_rows}) does not match output dataset rows ({output_row_count}).")
        except Exception as e:
            failures.append(f"Failed to load or verify TF-IDF matrix: {e}")

    # 4. Check TF-IDF Vectorizer Integrity
    if os.path.exists(tfidf_vec_path):
        try:
            vec = joblib.load(tfidf_vec_path)
            # Verify it is a scikit-learn TfidfVectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            if not isinstance(vec, TfidfVectorizer):
                failures.append(f"TF-IDF vectorizer at {paths_to_check['tfidf_vectorizer']} is not a TfidfVectorizer! Type: {type(vec)}")
        except Exception as e:
            failures.append(f"Failed to load or verify TF-IDF vectorizer: {e}")

    # 5. Verify hash metadata file matches actual hashes
    hash_meta_path = os.path.join(project_root, paths_to_check["hash"])
    if os.path.exists(hash_meta_path):
        try:
            with open(hash_meta_path, "r", encoding="utf-8") as f:
                hash_data = json.load(f)
            
            # Verify each logged file hash
            for fname, expected_hash in hash_meta_path.items() if isinstance(hash_meta_path, dict) else []:
                pass # (we will iterate over files we track in the hash list)
            
            files_to_hash = {
                "feature_dataset_v1.csv": output_path,
                "tfidf_vectorizer.joblib": tfidf_vec_path,
                "tfidf_matrix.joblib": tfidf_matrix_path
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

    # 6. Verify version metadata exists
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
                    failures.append("Last logged version does not reference the correct feature engineering output path.")
        except Exception as e:
            failures.append(f"Failed to verify version file integrity: {str(e)}")

    return len(failures) == 0, failures

if __name__ == "__main__":
    project_root_dir = os.path.dirname(os.path.abspath(__file__))
    passed, errors = verify_feature_integrity(project_root=project_root_dir)
    
    if passed:
        print("\n" + "="*50)
        print("PHASE 4 FEATURE ENGINEERING VERIFICATION PASSED")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("PHASE 4 FEATURE ENGINEERING VERIFICATION FAILED")
        print("="*50)
        for i, err in enumerate(errors, 1):
            print(f"{i}. {err}")
        print("="*50 + "\n")
        sys.exit(1)
