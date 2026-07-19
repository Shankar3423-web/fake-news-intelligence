import os
import sys
import time
import json
import hashlib
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

from ml.preprocessing.preprocessing_config import PreprocessingConfig
from ml.preprocessing.preprocessing_logger import setup_logger
from ml.preprocessing.preprocessing_validator import PreprocessingValidator
from ml.preprocessing.preprocessing_statistics import PreprocessingStatistics
from ml.preprocessing.preprocessing_profiler import PreprocessingProfiler
from ml.preprocessing.text_cleaner import TextCleaner
from ml.preprocessing.language_detector import LanguageDetector
from ml.preprocessing.tokenizer import Tokenizer
from ml.preprocessing.stopword_remover import StopwordRemover
from ml.preprocessing.lemmatizer import Lemmatizer
from ml.preprocessing.short_word_remover import ShortWordRemover
from ml.preprocessing.preprocessing_utils import ensure_nltk_resources, ensure_spacy_model

def run_preprocessing_pipeline(config_path: str = "config/preprocessing_config.yaml") -> bool:
    """
    Orchestrates the entire Phase 3 NLP Preprocessing pipeline.
    """
    start_time = time.time()
    
    # 1. Load config
    config = PreprocessingConfig(config_path)
    
    # 2. Setup logger
    logs_dir = config.get_path("logs_dir")
    logger = setup_logger(logs_dir)
    logger.info("Starting Phase 3 NLP Preprocessing Pipeline...")

    # 3. Validate input dataset
    input_path = config.get_path("input_dataset")
    validator = PreprocessingValidator()
    
    logger.info(f"Validating input dataset at: {input_path}")
    is_input_valid, input_errors = validator.validate_input(input_path)
    if not is_input_valid:
        logger.critical(f"Input dataset validation failed: {input_errors}")
        return False
    logger.info("Input dataset validation passed.")

    # 4. Download / verify NLP resources
    logger.info("Verifying NLP resource availability...")
    ensure_nltk_resources()
    spacy_ready = ensure_spacy_model(config.spacy_model)
    if not spacy_ready:
        logger.critical(f"Required spaCy model '{config.spacy_model}' could not be loaded or installed.")
        return False
    logger.info("NLP resources are ready.")

    # 5. Initialize processing components
    text_cleaner = TextCleaner(config)
    language_detector = LanguageDetector(
        supported_languages=config.supported_languages,
        default_language=config.default_language,
        fallback_on_error=config.fallback_on_error
    )
    tokenizer = Tokenizer()
    stopword_remover = StopwordRemover(
        language=config.stopword_language,
        custom_stopwords=config.custom_stopwords
    )
    lemmatizer = Lemmatizer(config.spacy_model)
    short_word_remover = ShortWordRemover(config.min_token_length)
    
    stats = PreprocessingStatistics()
    profiler = PreprocessingProfiler()

    # Ensure output directories exist
    os.makedirs(config.get_path("processed_dir"), exist_ok=True)
    os.makedirs(config.get_path("reports_dir"), exist_ok=True)
    os.makedirs(config.get_path("statistics_dir"), exist_ok=True)
    os.makedirs(config.get_path("versions_dir"), exist_ok=True)

    output_path = config.get_path("output_dataset")
    batch_size = config.batch_size
    
    logger.info(f"Beginning batch processing. Batch size: {batch_size}")
    
    # Read the dataset in chunks to optimize memory
    first_chunk = True
    input_row_count = 0
    rejected_rows_count = 0
    
    try:
        # Determine total rows for progress tracking if possible
        total_rows = 0
        try:
            # Quick count of rows in CSV
            with open(input_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1
            logger.info(f"Total rows to process: {total_rows}")
        except Exception:
            logger.warning("Could not determine total row count beforehand.")

        chunk_index = 0
        for chunk in pd.read_csv(input_path, chunksize=batch_size):
            chunk_index += 1
            logger.info(f"Processing batch {chunk_index}...")
            
            processed_rows = []
            
            # Reset local cleaner stats for this batch to update statistics accurately
            text_cleaner.reset()
            stopword_remover.reset()
            short_word_remover.reset()
            
            for idx, row in chunk.iterrows():
                input_row_count += 1
                original_text = row["text"]
                
                # Check for null/empty original text
                if pd.isna(original_text) or not str(original_text).strip():
                    stats.rows_failed += 1
                    logger.warning(f"Row ID {row.get('id', 'unknown')} has empty original text. Skipping.")
                    continue
                
                original_text_str = str(original_text)
                
                # Step 2-14: Text Cleaning
                try:
                    cleaned_str = text_cleaner.clean(original_text_str)
                except Exception as e:
                    stats.rows_failed += 1
                    logger.error(f"Text cleaning failed for Row ID {row.get('id')}: {e}")
                    continue
                
                # Step 15: Language Detection
                try:
                    lang = language_detector.detect(cleaned_str)
                    row_lang = lang
                except Exception as e:
                    logger.error(f"Language detection failed for Row ID {row.get('id')}: {e}")
                    stats.rows_failed += 1
                    continue
                
                if not language_detector.is_supported(lang):
                    rejected_rows_count += 1
                    logger.debug(f"Row ID {row.get('id')} rejected: unsupported language '{lang}'.")
                    continue
                
                # Step 16: Tokenization
                tokens = tokenizer.tokenize(cleaned_str)
                
                # Step 17: Stopword Removal
                tokens = stopword_remover.remove(tokens)
                
                # We collect intermediate values for batch lemmatization
                processed_rows.append({
                    "row_data": row.copy(),
                    "original_text": original_text_str,
                    "cleaned_str": cleaned_str,
                    "tokens": tokens,
                    "language": row_lang
                })

            if not processed_rows:
                logger.warning(f"No valid rows in batch {chunk_index} after cleaning and language check.")
                continue

            # Step 18: Lemmatization (using optimized batch processing)
            batch_tokens_list = [r["tokens"] for r in processed_rows]
            try:
                batch_lemmas = lemmatizer.lemmatize_batch(batch_tokens_list)
            except Exception as e:
                logger.error(f"Batch lemmatization failed: {e}. Falling back to single-row lemmatization.")
                batch_lemmas = [lemmatizer.lemmatize(t) for t in batch_tokens_list]

            # Step 19-20: Short word removal & Final clean text creation
            final_batch_rows = []
            for i, r in enumerate(processed_rows):
                lemmas = batch_lemmas[i]
                
                # Step 19: Short Word Removal
                lemmas = short_word_remover.remove(lemmas)
                
                # Step 20: Create Final Clean Text
                final_clean_text = " ".join(lemmas)
                
                # If final text is empty, check if we should keep it or write placeholder
                if not final_clean_text.strip():
                    # If empty, let's write a small placeholder or skip.
                    # Standard requirement is "No empty cleaned_text". So let's skip to keep output clean.
                    stats.rows_failed += 1
                    logger.warning(f"Row ID {r['row_data'].get('id')} became empty after cleaning/stopwords/lemmas. Skipping.")
                    continue
                
                # Update row data
                updated_row = r["row_data"]
                updated_row["cleaned_text"] = final_clean_text
                updated_row["language"] = r["language"]
                
                final_batch_rows.append(updated_row)
                
                # Update statistics and profiling
                stats.update_row_metrics(r["original_text"], final_clean_text, len(lemmas))
                profiler.update(r["tokens"], lemmas, r["language"])

            # Accumulate batch-level cleaner stats
            stats.update_cleaner_stats(text_cleaner.get_stats())
            stats.stopwords_removed += stopword_remover.removed_count
            
            # Write batch to output file
            if final_batch_rows:
                batch_df = pd.DataFrame(final_batch_rows)
                mode = "w" if first_chunk else "a"
                header = first_chunk
                batch_df.to_csv(output_path, mode=mode, index=False, header=header)
                first_chunk = False
                
            logger.info(f"Batch {chunk_index} completed. Running total processed: {stats.rows_processed}")

    except Exception as e:
        logger.critical(f"Critical pipeline failure: {e}", exc_info=True)
        return False

    # 6. Post-processing calculations and file generation
    end_time = time.time()
    execution_time = end_time - start_time
    stats.execution_time = execution_time
    stats.languages = language_detector.get_report()

    logger.info("Pipeline processing complete. Writing reports...")
    
    # Save statistics and profile
    stats.save(config.get_path("statistics_file"))
    profiler.save(config.get_path("profile_file"))

    # Compute SHA-256 Hash of output dataset
    logger.info("Generating dataset SHA-256 hash...")
    sha256_hash = hashlib.sha256()
    try:
        with open(output_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        digest = sha256_hash.hexdigest()
        
        hash_file_path = config.get_path("hash_file")
        with open(hash_file_path, "w", encoding="utf-8") as f:
            json.dump({"SHA256": digest}, f, indent=4)
        logger.info(f"Dataset hash saved to {hash_file_path}: {digest}")
    except Exception as e:
        logger.error(f"Failed to generate output dataset hash: {e}")
        digest = "N/A"

    # Save / Update Versioning metadata
    logger.info("Updating versioning database...")
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
        "Rows Processed": stats.rows_processed,
        "Execution Time": round(execution_time, 2),
        "Configuration Version": "1.0.0",
        "SHA256": digest
    }
    versions_data.append(current_version)
    try:
        with open(versions_file_path, "w", encoding="utf-8") as f:
            json.dump(versions_data, f, indent=4)
        logger.info(f"Version metadata updated in {versions_file_path}")
    except Exception as e:
        logger.error(f"Failed to save version metadata: {e}")

    # Generate Quality Report MD
    logger.info("Generating quality report...")
    quality_report_path = config.get_path("quality_report_file")
    try:
        # Determine token reduction percentage
        word_reduction = 0.0
        if stats.total_words > 0:
            word_reduction = ((stats.total_words - stats.total_tokens) / stats.total_words) * 100

        report_md = f"""# NLP Preprocessing Pipeline Quality Report

## Executive Summary
This report summarizes the NLP text preprocessing quality for the master dataset. The pipeline converted raw, noisy article text into normalized, clean tokens suitable for subsequent feature engineering.

- **Pipeline Version**: {current_version["Version"]}
- **Runtime**: {round(execution_time, 2)} seconds
- **Dataset Hash**: `{digest}`

---

## Dataset Summaries

### Input Dataset Summary
- **Input Path**: `{input_path}`
- **Total Input Rows**: {input_row_count}

### Output Dataset Summary
- **Output Path**: `{output_path}`
- **Successfully Preprocessed Rows**: {stats.rows_processed}
- **Skipped/Failed Rows**: {stats.rows_failed}
- **Language-Rejected Rows**: {rejected_rows_count}

---

## Cleaning Effectiveness

### Removed Noise Counts
- **HTML Tags Removed**: {stats.html_tags_removed}
- **URLs Removed**: {stats.urls_removed}
- **Email Addresses Removed**: {stats.emails_removed}
- **Emojis Removed**: {stats.emoji_count}
- **Special Characters Removed**: {stats.special_character_count}
- **Stopwords Removed**: {stats.stopwords_removed}

### Token Reduction Analysis
- **Average Words Per Input Article**: {round(stats.total_words / stats.rows_processed, 2) if stats.rows_processed > 0 else 0.0}
- **Average Tokens Per Output Article**: {round(stats.total_tokens / stats.rows_processed, 2) if stats.rows_processed > 0 else 0.0}
- **Token Reduction Rate**: {round(word_reduction, 2)}%

---

## Language Distribution
The following table shows the detected language distribution of processed articles prior to language filtering:

| Language Code | Article Count |
| --- | --- |
"""
        for lcode, count in stats.languages.items():
            report_md += f"| {lcode} | {count} |\n"

        report_md += """
---

## Recommendations
1. **Model Adaptation**: The high token reduction rate (removal of stopwords, punctuation, and short words) is highly suited for sparse bag-of-words or TF-IDF models. For neural models, some stopwords or punctuation might need to be preserved by modifying the configurations in `config/preprocessing_config.yaml`.
2. **Batch Optimization**: If dataset sizes scale past 100k+ rows, consider tuning `pipeline.batch_size` in the config file depending on available RAM.

---
*Report generated automatically on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*"
        
        with open(quality_report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Quality report saved to {quality_report_path}")
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}")

    logger.info("Pipeline completed successfully.")
    return True

if __name__ == "__main__":
    success = run_preprocessing_pipeline()
    sys.exit(0 if success else 1)
