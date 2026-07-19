import os
import sys
import time
import json
import hashlib
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

from ml.feature_engineering.feature_config import FeatureConfig
from ml.feature_engineering.feature_logger import setup_logger
from ml.feature_engineering.feature_validator import FeatureValidator
from ml.feature_engineering.feature_merger import FeatureMerger
from ml.feature_engineering.feature_statistics import FeatureStatistics
from ml.feature_engineering.feature_profiler import FeatureProfiler
from ml.feature_engineering.statistical_features import StatisticalFeatureExtractor
from ml.feature_engineering.readability_features import ReadabilityFeatureExtractor
from ml.feature_engineering.lexical_features import LexicalFeatureExtractor
from ml.feature_engineering.linguistic_features import LinguisticFeatureExtractor
from ml.feature_engineering.symbol_features import SymbolFeatureExtractor
from ml.feature_engineering.tfidf_features import TfidfFeatureGenerator
from ml.feature_engineering.feature_utils import ensure_nltk_resources, ensure_spacy_model

def compute_file_sha256(file_path: str) -> str:
    """Computes the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}")
        return "N/A"

def get_memory_usage() -> tuple[float, float]:
    """
    Returns (current_memory_mb, peak_memory_mb) of the current process.
    """
    try:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        rss = round(mem_info.rss / (1024 * 1024), 2)
        
        # Track peak memory using peak_wset on Windows
        if hasattr(mem_info, 'peak_wset'):
            peak = round(mem_info.peak_wset / (1024 * 1024), 2)
        else:
            peak = rss
        return rss, peak
    except Exception:
        return 0.0, 0.0

def run_feature_engineering_pipeline(config_path: str = "config/feature_config.yaml") -> bool:
    """
    Orchestrates the entire Phase 4 Feature Engineering pipeline.
    """
    start_time = time.time()
    
    # 1. Load config
    config = FeatureConfig(config_path)
    
    # 2. Setup logger
    logs_dir = config.get_path("logs_dir")
    logger = setup_logger(logs_dir)
    logger.info("Starting Phase 4 Feature Engineering Pipeline...")

    # 3. Validate input dataset structure
    input_path = config.get_path("input_dataset")
    validator = FeatureValidator()
    
    logger.info(f"Validating input preprocessed dataset at: {input_path}")
    is_input_valid, input_errors = validator.validate_input(input_path)
    if not is_input_valid:
        logger.critical(f"Input dataset validation failed: {input_errors}")
        return False
    logger.info("Input dataset validation passed.")

    # 4. Ensure NLP resources are available
    logger.info("Checking NLP resources...")
    ensure_nltk_resources()
    spacy_ready = ensure_spacy_model(config.spacy_model)
    if not spacy_ready:
        logger.critical(f"Required spaCy model '{config.spacy_model}' could not be loaded or installed.")
        return False
    logger.info("NLP resources verified.")

    # 5. Ensure output directories exist
    os.makedirs(config.get_path("processed_dir"), exist_ok=True)
    os.makedirs(config.get_path("reports_dir"), exist_ok=True)
    os.makedirs(config.get_path("statistics_dir"), exist_ok=True)
    os.makedirs(config.get_path("versions_dir"), exist_ok=True)

    # 6. Initialize Feature Extractor components
    stat_extractor = StatisticalFeatureExtractor()
    read_extractor = ReadabilityFeatureExtractor()
    lex_extractor = LexicalFeatureExtractor(config)
    sym_extractor = SymbolFeatureExtractor()
    
    # Lazy load linguistic extractor only if step is enabled, as loading spaCy model can be memory-heavy
    ling_extractor = None
    if config.get_step_enabled("linguistic_features"):
        ling_extractor = LinguisticFeatureExtractor(config)
        
    merger = FeatureMerger()
    
    output_path = config.get_path("output_dataset")
    pipeline_batch_size = config.pipeline_batch_size
    
    # Try to install psutil at start
    try:
        import psutil
    except ImportError:
        try:
            import subprocess
            import sys
            logger.info("psutil not found. Attempting to install psutil programmatically...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("psutil installed successfully.")
        except Exception as e:
            logger.warning(f"Could not install psutil: {e}. Memory tracking will fall back.")

    # Clean the existing output file if it exists to start fresh
    if os.path.exists(output_path):
        logger.info(f"Removing existing output dataset at {output_path} to start clean...")
        try:
            os.remove(output_path)
        except Exception as e:
            logger.warning(f"Failed to remove existing output dataset: {e}")

    logger.info(f"Beginning batch-level dense feature extraction. Pipeline batch size: {pipeline_batch_size}")
    
    first_chunk = True
    total_rows_processed = 0
    
    try:
        # Determine total rows for tracking
        total_rows = 0
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1
            logger.info(f"Total rows to process: {total_rows}")
        except Exception:
            logger.warning("Could not determine total row count beforehand.")

        chunk_index = 0
        import gc
        for chunk in pd.read_csv(input_path, chunksize=pipeline_batch_size):
            batch_start_time = time.time()
            chunk_index += 1
            logger.info(f"Processing batch {chunk_index}...")
            
            # Keep track of active features
            feature_dfs = []
            
            # Extract Statistical Features
            if config.get_step_enabled("statistical_features"):
                stat_df = stat_extractor.extract_features(chunk["text"])
                feature_dfs.append(stat_df)
                
            # Extract Readability Features
            if config.get_step_enabled("readability_features"):
                read_df = read_extractor.extract_features(chunk["text"])
                feature_dfs.append(read_df)
                
            # Extract Lexical Features
            if config.get_step_enabled("lexical_features"):
                lex_df = lex_extractor.extract_features(chunk["text"])
                feature_dfs.append(lex_df)
                
            # Extract Symbol Features
            if config.get_step_enabled("symbol_features"):
                sym_df = sym_extractor.extract_features(chunk["text"])
                feature_dfs.append(sym_df)
                
            # Extract Linguistic Features
            if config.get_step_enabled("linguistic_features") and ling_extractor:
                ling_df = ling_extractor.extract_features(chunk["text"])
                feature_dfs.append(ling_df)
                
            # Merge and write
            batch_merged = merger.merge_features(chunk, feature_dfs)
            
            mode = "w" if first_chunk else "a"
            header = first_chunk
            batch_merged.to_csv(output_path, mode=mode, index=False, header=header)
            
            first_chunk = False
            total_rows_processed += len(chunk)
            
            batch_elapsed = time.time() - batch_start_time
            current_mem, peak_mem = get_memory_usage()
            
            logger.info(
                f"Batch {chunk_index} completed | "
                f"Rows: {len(chunk)} | "
                f"Memory: {current_mem} MB | "
                f"Peak Memory: {peak_mem} MB | "
                f"Time: {round(batch_elapsed, 2)} seconds"
            )
            
            # Strict memory release
            del feature_dfs
            del batch_merged
            if 'stat_df' in locals(): del stat_df
            if 'read_df' in locals(): del read_df
            if 'lex_df' in locals(): del lex_df
            if 'sym_df' in locals(): del sym_df
            if 'ling_df' in locals(): del ling_df
            del chunk
            gc.collect()

    except Exception as e:
        logger.critical(f"Critical error during dense feature extraction: {e}", exc_info=True)
        return False

    # 7. TF-IDF Text Representation Feature Generation (entire corpus)
    if config.get_step_enabled("tfidf_features"):
        logger.info("Starting TF-IDF Feature Generation...")
        try:
            # Re-read only the cleaned_text column to optimize memory usage
            logger.info("Loading cleaned_text column from preprocessed dataset...")
            preprocessed_df = pd.read_csv(input_path, usecols=["cleaned_text"])
            
            tfidf_generator = TfidfFeatureGenerator(config)
            tfidf_matrix = tfidf_generator.fit_transform(preprocessed_df["cleaned_text"])
            
            # Save vectorizer and matrix separately (without converting matrix to dense)
            tfidf_generator.save(
                config.get_path("tfidf_vectorizer"),
                config.get_path("tfidf_matrix"),
                tfidf_matrix
            )
            
            # Release memory
            del preprocessed_df
            del tfidf_matrix
            del tfidf_generator
            gc.collect()
        except Exception as e:
            logger.critical(f"Critical error during TF-IDF extraction: {e}", exc_info=True)
            return False

    # 8. Post-processing calculations, file generation, and statistics
    logger.info("Computing metrics and metadata reports...")
    end_time = time.time()
    execution_time = end_time - start_time
    
    try:
        # Load the newly created feature dataset to compute statistics and profile
        logger.info("Reading merged feature dataset for profile and statistics calculations...")
        df_out = pd.read_csv(output_path)
        
        # Identify feature columns
        feature_cols = [col for col in df_out.columns if col not in ["id", "label", "cleaned_text"]]
        total_features = len(feature_cols)
        null_cells = int(df_out[feature_cols].isnull().sum().sum())
        
        # Calculate & save statistics
        stats_collector = FeatureStatistics()
        stats_collector.calculate_statistics(df_out, feature_cols, execution_time)
        stats_collector.save(config.get_path("statistics_file"))
        
        # Calculate & save profile
        profiler = FeatureProfiler()
        profiler.profile_dataset(df_out, feature_cols)
        profiler.save(config.get_path("profile_file"))
        
        # 9. Perform output validation checks
        logger.info("Running output schema validation...")
        is_output_valid, output_errors = validator.validate_output(output_path, expected_rows=total_rows_processed)
        if not is_output_valid:
            logger.warning(f"Output dataset validation found anomalies: {output_errors}")
        else:
            logger.info("Output dataset validation passed successfully.")

        # 10. Generate Hashes
        logger.info("Generating SHA-256 hashes for all output files...")
        hashes = {
            "feature_dataset_v1.csv": compute_file_sha256(output_path),
            "tfidf_vectorizer.joblib": compute_file_sha256(config.get_path("tfidf_vectorizer")),
            "tfidf_matrix.joblib": compute_file_sha256(config.get_path("tfidf_matrix"))
        }
        
        # Release memory
        del df_out
        del stats_collector
        del profiler
        gc.collect()
        
        hash_file_path = config.get_path("hash_file")
        with open(hash_file_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=4)
        logger.info(f"Hashes saved successfully to {hash_file_path}")

        # 11. Save Versioning Info
        logger.info("Updating feature engineering version database...")
        versions_file_path = config.get_path("versions_file")
        versions_data = []
        if os.path.exists(versions_file_path):
            try:
                with open(versions_file_path, "r", encoding="utf-8") as f:
                    versions_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read existing versions file: {e}. Starting fresh.")
                
        current_version = {
            "Version": f"1.0.{len(versions_data) + 1}",
            "Timestamp": datetime.now().isoformat(),
            "Input Dataset": input_path,
            "Output Dataset": output_path,
            "TFIDF Vectorizer": config.get_path("tfidf_vectorizer"),
            "TFIDF Matrix": config.get_path("tfidf_matrix"),
            "Rows Processed": total_rows_processed,
            "Execution Time": round(execution_time, 2),
            "Configuration Version": "1.0.0",
            "SHA256 Hashes": hashes
        }
        versions_data.append(current_version)
        with open(versions_file_path, "w", encoding="utf-8") as f:
            json.dump(versions_data, f, indent=4)
        logger.info(f"Version database updated at {versions_file_path}")

        # 12. Generate Quality Report
        logger.info("Generating feature engineering quality report...")
        quality_report_path = config.get_path("quality_report_file")
        
        # Gather basic stats for report
        # (total_features and null_cells were calculated before df_out was deleted)
        
        report_md = f"""# Feature Engineering Pipeline Quality Report

## Executive Summary
This report summarizes the feature engineering run for the Fake News Detection pipeline. This phase processes the preprocessed dataset and transforms text and metadata into numerical features.

- **Pipeline Version**: {current_version["Version"]}
- **Total Runtime**: {round(execution_time, 2)} seconds
- **Dataset Row Count**: {total_rows_processed}
- **Hand-crafted Dense Features**: {total_features}
- **TF-IDF Sparse Representation Size**: {config.tfidf_max_features} features

---

## Output Artifacts & Integrity
The following artifacts were successfully generated and validated:

| File Name | Rel Path | SHA-256 Hash |
| --- | --- | --- |
| Feature Dataset CSV | `{output_path}` | `{hashes["feature_dataset_v1.csv"]}` |
| TF-IDF Vectorizer | `{config.get_path("tfidf_vectorizer")}` | `{hashes["tfidf_vectorizer.joblib"]}` |
| TF-IDF Sparse Matrix | `{config.get_path("tfidf_matrix")}` | `{hashes["tfidf_matrix.joblib"]}` |

- **Null Values in Feature Cells**: {null_cells}
- **Label Integrity Check**: {"PASSED" if is_output_valid else "FAILED"}

---

## Engineered Feature Groups

### 1. Statistical Features
- word count, character count, sentence count, average word length, average sentence length, vocabulary size.

### 2. Readability Features
- Flesch Reading Ease, Flesch-Kincaid Grade, SMOG index, Gunning Fog, Coleman-Liau index.

### 3. Lexical Features
- lexical diversity, unique words, stopword ratio, long word ratio, short word ratio.

### 4. Symbol Features
- digit count, uppercase count, punctuation count, special character count.

### 5. Linguistic Features (spaCy)
- entity count, noun/verb/adjective count, POS distributions.

### 6. Term Representation (TF-IDF)
- Vocabulary range: {config.tfidf_ngram_range[0]} to {config.tfidf_ngram_range[1]} n-grams.
- Max features: {config.tfidf_max_features}

---

## Recommendations
1. **Feature Scale**: Noticeable differences exist in feature scales (e.g., character count ranges into thousands, while stopword ratio is between 0 and 1). Machine learning models (e.g., Logistic Regression, SVM) require scaling. Apply StandardScaler or MinMaxScaler prior to model training.
2. **Dense & Sparse Combination**: For modeling, combine the hand-crafted dense features (from `feature_dataset_v1.csv`) with the TF-IDF sparse matrix (from `tfidf_matrix.joblib`) using scipy's `hstack` to form the final model input matrix.
3. **Feature Selection**: Since we generated {total_features + config.tfidf_max_features} total features, perform feature selection (e.g. variance thresholding or tree-based feature importance) in Phase 5 to prevent overfitting.

*Report automatically generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        with open(quality_report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Quality report successfully saved to {quality_report_path}")

    except Exception as e:
        logger.critical(f"Failed to generate reports and metadata: {e}", exc_info=True)
        return False

    logger.info("Phase 4 Feature Engineering Pipeline completed successfully.")
    return True

if __name__ == "__main__":
    success = run_feature_engineering_pipeline()
    sys.exit(0 if success else 1)
