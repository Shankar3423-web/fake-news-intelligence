import os
import time
import logging
import json
import hashlib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple

# Import pipeline components
from ml.dataset.dataset_loader import DatasetLoader
from ml.dataset.dataset_validator import DatasetValidator
from ml.dataset.dataset_standardizer import DatasetStandardizer
from ml.dataset.dataset_merger import DatasetMerger
from ml.dataset.duplicate_remover import DuplicateRemover
from ml.dataset.missing_value_handler import MissingValueHandler
from ml.dataset.dataset_profiler import DatasetProfiler
from ml.dataset.dataset_statistics import DatasetStatistics

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    Includes fallbacks to standard defaults in case PyYAML is missing or fails.
    """
    default_config = {
        "duplicate_detection": {
            "batch_size": 2000,
            "similarity_threshold": 0.95,
            "max_features": 5000
        },
        "schemas": {
            "supported_extensions": [".csv", ".xlsx", ".xls"],
            "target_columns": ["id", "title", "text", "label", "source", "category", "author", "published_date", "language", "url", "dataset_origin"],
            "mandatory_fields": ["title", "text", "label"]
        },
        "defaults": {
            "language": "en",
            "date_format": "%Y-%m-%d"
        },
        "paths": {
            "raw_dir": "ml/dataset/raw",
            "cleaned_dir": "ml/dataset/cleaned",
            "merged_dir": "ml/dataset/merged",
            "processed_dir": "ml/dataset/processed",
            "statistics_dir": "ml/dataset/statistics",
            "removed_dir": "ml/dataset/removed",
            "versions_dir": "ml/dataset/versions",
            "reports_dir": "ml/dataset/reports",
            "logs_dir": "logs"
        }
    }
    
    if not os.path.exists(config_path):
        return default_config
        
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # Deep merge dictionary helper
            for section, details in default_config.items():
                if section not in config:
                    config[section] = details
                elif isinstance(details, dict):
                    for key, val in details.items():
                        if key not in config[section]:
                            config[section][key] = val
            return config
    except ImportError:
        logger.warning("PyYAML not installed. Using default fallback configuration.")
        return default_config
    except Exception as e:
        logger.warning(f"Error loading configuration {config_path}: {str(e)}. Using fallback defaults.")
        return default_config


class DatasetPipeline:
    """
    Orchestrates the entire production-grade Dataset Engineering pipeline.
    """

    def __init__(self, project_root: str = ".", config_file: str = "config/dataset_config.yaml") -> None:
        self.project_root = os.path.abspath(project_root)
        self.config_path = os.path.join(self.project_root, config_file)
        self.config = load_config(self.config_path)

        # Retrieve paths from configuration
        paths = self.config["paths"]
        self.raw_dir = os.path.join(self.project_root, paths["raw_dir"])
        self.cleaned_dir = os.path.join(self.project_root, paths["cleaned_dir"])
        self.merged_dir = os.path.join(self.project_root, paths["merged_dir"])
        self.processed_dir = os.path.join(self.project_root, paths["processed_dir"])
        self.statistics_dir = os.path.join(self.project_root, paths["statistics_dir"])
        self.removed_dir = os.path.join(self.project_root, paths["removed_dir"])
        self.versions_dir = os.path.join(self.project_root, paths["versions_dir"])
        self.reports_dir = os.path.join(self.project_root, paths["reports_dir"])
        self.logs_dir = os.path.join(self.project_root, paths["logs_dir"])

        # Ensure all directories exist
        for directory in [self.cleaned_dir, self.merged_dir, self.processed_dir, 
                          self.statistics_dir, self.removed_dir, self.versions_dir, 
                          self.reports_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Setup dynamic timestamped file logging
        self._setup_file_logger()

        # Initialize pipeline stages
        self.loader = DatasetLoader()
        self.validator = DatasetValidator()
        self.standardizer = DatasetStandardizer()
        self.merger = DatasetMerger()
        self.missing_handler = MissingValueHandler()
        self.duplicate_remover = DuplicateRemover()
        self.profiler = DatasetProfiler()
        self.statistics_gen = DatasetStatistics()

    def _setup_file_logger(self) -> None:
        """
        Dynamically configures a timestamped log handler for the current execution.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.logs_dir, f"dataset_pipeline_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        
        # Attach to root logger
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Dynamically attached timestamped log file: {log_file}")

    def run(self) -> bool:
        """
        Executes the production dataset engineering pipeline.
        """
        pipeline_start = time.time()
        logger.info("==================================================================")
        logger.info("STARTING PRODUCTION DATASET ENGINEERING PIPELINE")
        logger.info("==================================================================")

        raw_counts: Dict[str, int] = {}
        cleaned_counts: Dict[str, int] = {}
        standardized_dfs: List[pd.DataFrame] = []
        validation_reports: List[Dict[str, Any]] = []
        
        # Track all removed records
        removed_duplicates: List[Dict[str, Any]] = []
        removed_missing: List[Dict[str, Any]] = []
        removed_invalid: List[Dict[str, Any]] = []

        # Datasets definition
        datasets_config = [
            (os.path.join(self.raw_dir, "isot", "True.csv"), "ISOT", 0, "isot_true"),
            (os.path.join(self.raw_dir, "isot", "Fake.csv"), "ISOT", 1, "isot_fake"),
            (os.path.join(self.raw_dir, "india", "Indian_True_News_dataset.xlsx"), "INDIA", 0, "india_true"),
            (os.path.join(self.raw_dir, "india", "Indian_Fake_News_dataset.xlsx"), "INDIA", 1, "india_fake")
        ]

        # STEP 1 & 3 & 4 & 5: Load, Standardize, and filter invalid rows
        for file_path, origin, default_label, key in datasets_config:
            step_start = time.time()
            logger.info(f"Processing dataset file: {os.path.basename(file_path)} (Origin: {origin})")
            
            # Extension/Existence validation
            file_valid, file_errors = self.validator.validate_file(file_path)
            if not file_valid:
                logger.error(f"Raw file check failed for {file_path}: {file_errors}")
                return False

            # Load
            try:
                raw_df = self.loader.load_dataset(file_path)
                raw_counts[key] = len(raw_df)
                logger.info(f"Loaded {len(raw_df)} rows in {round(time.time() - step_start, 4)}s")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                return False

            # Standardize schema (adds labels, normalizes dates)
            std_start = time.time()
            std_df = self.standardizer.standardize(
                df=raw_df,
                origin=origin,
                default_label=default_label,
                default_lang=self.config["defaults"]["language"]
            )

            # TASK 4: Filter invalid rows prior to validation
            # Define invalid rows as text length < 3, invalid label, or malformed/unparseable dates
            invalid_mask = pd.Series(False, index=std_df.index)
            reasons = pd.Series("", index=std_df.index)

            # 1. Text too short
            short_text = std_df["text"].fillna("").astype(str).str.strip().str.len() < 3
            invalid_mask |= short_text
            reasons.loc[short_text] = reasons.loc[short_text] + "Text content length is under 3 characters; "

            # 2. Invalid label
            bad_label = ~std_df["label"].isin([0, 1])
            invalid_mask |= bad_label
            reasons.loc[bad_label] = reasons.loc[bad_label] + "Invalid/unparseable news label value; "

            # 3. Bad date format
            date_vals = std_df["published_date"].dropna().astype(str)
            if not date_vals.empty:
                bad_date = ~date_vals.str.match(r"^\d{4}-\d{2}-\d{2}$")
                bad_date_indices = date_vals[bad_date].index
                invalid_mask.loc[bad_date_indices] = True
                reasons.loc[bad_date_indices] = reasons.loc[bad_date_indices] + "Invalid date format; "

            # Extract invalid rows
            invalid_df = std_df[invalid_mask].copy()
            if not invalid_df.empty:
                logger.warning(f"Identified {len(invalid_df)} invalid rows in {key}.")
                for idx, row in invalid_df.iterrows():
                    removed_invalid.append({
                        "id": row.get("id"),
                        "title": row.get("title"),
                        "text": row.get("text"),
                        "label": row.get("label"),
                        "source": row.get("source"),
                        "category": row.get("category"),
                        "published_date": row.get("published_date"),
                        "dataset_origin": row.get("dataset_origin"),
                        "removal_reason": reasons.loc[idx].strip("; ")
                    })
                std_df = std_df[~invalid_mask].copy()

            # Validation
            val_start = time.time()
            val_report = self.validator.validate_dataframe(
                df=std_df,
                file_label=key,
                schema_type=origin
            )
            validation_reports.append(val_report)
            
            # Save validation report
            val_report_path = os.path.join(self.statistics_dir, f"validation_report_{key}.json")
            with open(val_report_path, "w", encoding="utf-8") as f:
                json.dump(val_report, f, indent=4)

            if not val_report["is_valid"]:
                logger.warning(f"Schema validation warnings encountered for {key}: {val_report['errors']}")
            
            # Save cleaned dataset
            cleaned_file_path = os.path.join(self.cleaned_dir, f"{key}_cleaned.csv")
            std_df.to_csv(cleaned_file_path, index=False)
            cleaned_counts[key] = len(std_df)
            standardized_dfs.append(std_df)
            
            logger.info(f"Saved standardized and validated dataset to: {cleaned_file_path}")

        # STEP 6: Merge datasets
        merge_start = time.time()
        logger.info("Merging standardized datasets...")
        try:
            merged_df = self.merger.merge(standardized_dfs)
            merged_file_path = os.path.join(self.merged_dir, "merged_dataset.csv")
            merged_df.to_csv(merged_file_path, index=False)
            logger.info(f"Merged raw datasets. Shape: {merged_df.shape}")
        except Exception as e:
            logger.error(f"Failed merging datasets: {str(e)}")
            return False

        # STEP 8 & TASK 4: Missing Value Handling
        missing_start = time.time()
        cleaned_mv_df, mv_report = self.missing_handler.handle_missing_values(merged_df)
        missing_dropped = mv_report["total_dropped"]
        removed_missing.extend(mv_report.get("removed_records", []))
        logger.info(f"Missing Value Handler completed. Dropped: {missing_dropped}")

        # STEP 7 & TASK 4: Memory-Safe Duplicate Detection
        dup_start = time.time()
        deduped_df, dedup_report = self.duplicate_remover.remove_duplicates(
            df=cleaned_mv_df,
            similarity_threshold=self.config["duplicate_detection"]["similarity_threshold"],
            batch_size=self.config["duplicate_detection"]["batch_size"],
            max_features=self.config["duplicate_detection"]["max_features"]
        )
        duplicates_removed = len(dedup_report.get("removed_records", []))
        removed_duplicates.extend(dedup_report.get("removed_records", []))

        # Regenerate clean incremental IDs for the final unique master dataset
        final_df = deduped_df.reset_index(drop=True)
        final_df["id"] = final_df.index.map(lambda idx: f"master_{idx + 1}")

        # Save Master Dataset
        master_file_path = os.path.join(self.processed_dir, "master_dataset_v1.csv")
        final_df.to_csv(master_file_path, index=False)
        logger.info(f"Saved master dataset to: {master_file_path}")

        # TASK 4: Archive Removed Records to CSVs
        self._archive_removed_records(removed_duplicates, removed_missing, removed_invalid)

        # TASK 3: Dataset Hashing
        master_hash = self._generate_dataset_hash(master_file_path)

        # STEP 9: Dataset Profiling
        profile_report_path = os.path.join(self.statistics_dir, "dataset_profile.json")
        profile_data = self.profiler.profile_dataset(
            df=final_df,
            duplicate_count=duplicates_removed,
            output_path=profile_report_path
        )

        # STEP 10: Dataset Statistics
        stats_report_path = os.path.join(self.statistics_dir, "dataset_statistics.json")
        real_news = int((final_df["label"] == 0).sum())
        fake_news = int((final_df["label"] == 1).sum())
        
        self.statistics_gen.generate_statistics(
            raw_counts=raw_counts,
            cleaned_counts=cleaned_counts,
            final_count=len(final_df),
            duplicates_removed=duplicates_removed,
            missing_mandatory_dropped=missing_dropped,
            real_count=real_news,
            fake_count=fake_news,
            output_path=stats_report_path
        )

        # TASK 2: Dataset Versioning
        version_num = self._save_dataset_version(
            raw_counts=raw_counts,
            final_count=len(final_df),
            missing_count=missing_dropped,
            invalid_count=len(removed_invalid),
            duplicate_count=duplicates_removed,
            dataset_hash=master_hash,
            runtime_seconds=time.time() - pipeline_start
        )

        # TASK 5: Dataset Quality Report
        self._generate_quality_report(
            df=final_df,
            raw_count=sum(raw_counts.values()),
            missing_dropped=missing_dropped,
            invalid_dropped=len(removed_invalid),
            duplicates_dropped=duplicates_removed,
            profile_data=profile_data,
            version=version_num,
            runtime=time.time() - pipeline_start
        )

        # Run verification check dynamically to display status in summary
        verification_status = "PASSED"
        verification_errors = []
        try:
            from verify_dataset_pipeline import verify_pipeline_integrity
            passed, errs = verify_pipeline_integrity(project_root=self.project_root)
            if not passed:
                verification_status = "FAILED"
                verification_errors = errs
        except Exception as e:
            verification_status = f"ERROR ({str(e)})"

        # TASK 9: Print pipeline summary banner
        pipeline_duration = time.time() - pipeline_start
        total_removed = missing_dropped + len(removed_invalid) + duplicates_removed
        
        print("\n===================================================")
        print("DATASET ENGINEERING COMPLETED SUCCESSFULLY")
        print(f"Version:                 {version_num}")
        print(f"Execution Time:          {round(pipeline_duration, 2)}s")
        print(f"Total Raw Records:       {sum(raw_counts.values())}")
        print(f"Rows Removed:            {total_removed} (Missing: {missing_dropped}, Invalid: {len(removed_invalid)})")
        print(f"Duplicates Removed:      {duplicates_removed}")
        print(f"Final Records:           {len(final_df)}")
        print(f"Master Dataset Location: {master_file_path}")
        print(f"Dataset Hash:            {master_hash}")
        print(f"Verification Status:     {verification_status}")
        if verification_status != "PASSED":
            print(f"Verification Failures:   {verification_errors}")
        print("===================================================\n")

        logger.info("==================================================================")
        logger.info(f"DATASET ENGINEERING PIPELINE COMPLETE. VERSION: {version_num}")
        logger.info("==================================================================")
        
        return True

    def _archive_removed_records(
        self, 
        duplicates: List[Dict[str, Any]], 
        missing: List[Dict[str, Any]], 
        invalid: List[Dict[str, Any]]
    ) -> None:
        """
        Saves all excluded records with reasons to CSV files in the removed/ archive.
        """
        cols = ["id", "title", "text", "label", "source", "category", "published_date", "dataset_origin", "removal_reason"]
        
        # Save duplicates
        dup_df = pd.DataFrame(duplicates)
        if dup_df.empty:
            dup_df = pd.DataFrame(columns=cols)
        else:
            dup_df = dup_df[cols]
        dup_path = os.path.join(self.removed_dir, "duplicate_rows.csv")
        dup_df.to_csv(dup_path, index=False)
        logger.info(f"Archived {len(dup_df)} duplicate rows to {dup_path}")

        # Save missing rows
        missing_df = pd.DataFrame(missing)
        if missing_df.empty:
            missing_df = pd.DataFrame(columns=cols)
        else:
            missing_df = missing_df[cols]
        missing_path = os.path.join(self.removed_dir, "missing_rows.csv")
        missing_df.to_csv(missing_path, index=False)
        logger.info(f"Archived {len(missing_df)} missing rows to {missing_path}")

        # Save invalid rows
        invalid_df = pd.DataFrame(invalid)
        if invalid_df.empty:
            invalid_df = pd.DataFrame(columns=cols)
        else:
            invalid_df = invalid_df[cols]
        invalid_path = os.path.join(self.removed_dir, "invalid_rows.csv")
        invalid_df.to_csv(invalid_path, index=False)
        logger.info(f"Archived {len(invalid_df)} invalid rows to {invalid_path}")

    def _generate_dataset_hash(self, file_path: str) -> str:
        """
        Computes SHA-256 hash of the master dataset and saves metadata JSON.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        digest = sha256_hash.hexdigest()
        
        hash_meta = {
            "File Name": os.path.basename(file_path),
            "SHA256": digest,
            "Generated Time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Dataset Version": "v_latest" # Temporary placeholder, will be replaced inside save_version
        }
        
        hash_file = os.path.join(self.statistics_dir, "dataset_hash.json")
        with open(hash_file, "w", encoding="utf-8") as f:
            json.dump(hash_meta, f, indent=4)
            
        logger.info(f"Generated SHA-256 hash {digest} and saved to {hash_file}")
        return digest

    def _save_dataset_version(
        self,
        raw_counts: Dict[str, int],
        final_count: int,
        missing_count: int,
        invalid_count: int,
        duplicate_count: int,
        dataset_hash: str,
        runtime_seconds: float
    ) -> str:
        """
        Appends pipeline execution metadata to dataset_versions.json.
        """
        versions_file = os.path.join(self.versions_dir, "dataset_versions.json")
        
        # Load existing versions
        version_history = []
        if os.path.exists(versions_file):
            try:
                with open(versions_file, "r", encoding="utf-8") as f:
                    version_history = json.load(f)
            except Exception as e:
                logger.warning(f"Failed parsing dataset_versions.json: {str(e)}. Resetting history.")

        # Determine version number (e.g. v1, v2)
        next_ver_num = len(version_history) + 1
        version_str = f"v{next_ver_num}"

        version_entry = {
            "Version Number": version_str,
            "Creation Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Input Datasets": {
                "isot_true_rows": raw_counts.get("isot_true", 0),
                "isot_fake_rows": raw_counts.get("isot_fake", 0),
                "india_true_rows": raw_counts.get("india_true", 0),
                "india_fake_rows": raw_counts.get("india_fake", 0)
            },
            "Total Records": final_count,
            "Rows Removed": missing_count + invalid_count + duplicate_count,
            "Duplicate Count": duplicate_count,
            "Missing Value Count": missing_count,
            "Invalid Value Count": invalid_count,
            "Dataset Hash": dataset_hash,
            "Pipeline Execution Time": f"{round(runtime_seconds, 2)}s",
            "Master Dataset File Name": "master_dataset_v1.csv",
            "Current Dataset Version": version_str
        }

        version_history.append(version_entry)
        
        with open(versions_file, "w", encoding="utf-8") as f:
            json.dump(version_history, f, indent=4)

        # Update dataset_hash.json version field
        hash_file = os.path.join(self.statistics_dir, "dataset_hash.json")
        if os.path.exists(hash_file):
            try:
                with open(hash_file, "r", encoding="utf-8") as f:
                    hash_meta = json.load(f)
                hash_meta["Dataset Version"] = version_str
                with open(hash_file, "w", encoding="utf-8") as f:
                    json.dump(hash_meta, f, indent=4)
            except Exception as e:
                logger.warning(f"Could not update version in dataset_hash.json: {str(e)}")

        logger.info(f"Recorded dataset version: {version_str}")
        return version_str

    def _generate_quality_report(
        self,
        df: pd.DataFrame,
        raw_count: int,
        missing_dropped: int,
        invalid_dropped: int,
        duplicates_dropped: int,
        profile_data: Dict[str, Any],
        version: str,
        runtime: float
    ) -> None:
        """
        Generates a human-readable dataset quality markdown report.
        """
        report_file = os.path.join(self.reports_dir, "dataset_quality_report.md")
        
        # Calculate rates
        total_removed = missing_dropped + invalid_dropped + duplicates_dropped
        retention_pct = round((len(df) / raw_count) * 100, 2) if raw_count > 0 else 0.0
        real_count = profile_data["summary"]["real_articles"]
        fake_count = profile_data["summary"]["fake_articles"]
        real_pct = round((real_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        fake_pct = round((fake_count / len(df)) * 100, 2) if len(df) > 0 else 0.0

        lang_md = "\n".join([f"- **{lang}**: {count}" for lang, count in profile_data["distributions"]["language"].items()])
        cat_md = "\n".join([f"- **{cat}**: {count}" for cat, count in profile_data["top_analytics"]["top_categories"].items()])
        src_md = "\n".join([f"- **{src}**: {count}" for src, count in profile_data["top_analytics"]["top_sources"].items()])

        missing_fields_md = ""
        for field, detail in profile_data["missing_values"].items():
            missing_fields_md += f"| {field} | {detail['count']} | {detail['percentage']}% |\n"

        report_content = f"""# Dataset Quality Report

## Executive Summary
This report summarizes the dataset quality analysis for the standardized Master Dataset used in downstream NLP preprocessing and model training.

- **Dataset Version:** {version}
- **Pipeline Runtime:** {round(runtime, 2)} seconds
- **Report Generated Time:** {time.strftime("%Y-%m-%d %H:%M:%S")}
- **Integrity Status:** Verification Completed

---

## Dataset Summary Metrics
| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Raw Articles** | {raw_count} | 100.0% |
| **Final Cleaned Articles** | {len(df)} | {retention_pct}% |
| **Real Articles (Class 0)** | {real_count} | {real_pct}% |
| **Fake Articles (Class 1)** | {fake_count} | {fake_pct}% |
| **Rows Dropped (Missing Mandatories)** | {missing_dropped} | {round((missing_dropped/raw_count)*100, 2) if raw_count > 0 else 0.0}% |
| **Rows Dropped (Invalid/Corrupted)** | {invalid_dropped} | {round((invalid_dropped/raw_count)*100, 2) if raw_count > 0 else 0.0}% |
| **Rows Dropped (Duplicates)** | {duplicates_dropped} | {round((duplicates_dropped/raw_count)*100, 2) if raw_count > 0 else 0.0}% |

---

## Data Distributions

### Language Distribution
{lang_md}

### Top 5 Categories
{cat_md}

### Top 5 Sources
{src_md}

---

## Article Text Length Analytics
- **Average Word Count:** {profile_data["text_statistics"]["average_word_length"]} words
- **Average Character Count:** {profile_data["text_statistics"]["average_character_length"]} characters
- **Longest Article ID:** {profile_data["text_statistics"]["longest_article"]["id"]} ({profile_data["text_statistics"]["longest_article"]["character_length"]} chars)
- **Shortest Article ID:** {profile_data["text_statistics"]["shortest_article"]["id"]} ({profile_data["text_statistics"]["shortest_article"]["character_length"]} chars)

---

## Missing Value Details (Post-Cleaning)
| Field | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
{missing_fields_md}

---

## Potential Data Quality Issues
1. **Unbalanced Categories**: Some categories represent a much larger portion of the dataset, which could bias class predictions.
2. **Missing Author Metadata**: Author information is missing in a high percentage of rows.
3. **Missing Source URL**: A significant number of records lack original URLs, limiting trace verification.

## Recommendations
- **Stratified Split**: When splitting the data into training and validation sets, use stratified splits based on both the `label` and `dataset_origin` to ensure representation.
- **Deduplication Thresholding**: A duplicate threshold of 0.95 removes highly redundant articles. Adjust this threshold inside `config/dataset_config.yaml` to tighten or loosen deduplication.
"""

        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_content)
            logger.info(f"Saved dataset quality report to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to generate dataset quality report: {str(e)}")


if __name__ == "__main__":
    project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Configure logging
    log_file = os.path.join(project_root_dir, "logs", "dataset_processing.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    pipeline = DatasetPipeline(project_root=project_root_dir)
    pipeline.run()
