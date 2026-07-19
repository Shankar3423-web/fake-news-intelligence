import os
import tempfile
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from ml.dataset.dataset_loader import DatasetLoader
from ml.dataset.dataset_validator import DatasetValidator
from ml.dataset.dataset_standardizer import DatasetStandardizer
from ml.dataset.dataset_merger import DatasetMerger
from ml.dataset.duplicate_remover import DuplicateRemover
from ml.dataset.missing_value_handler import MissingValueHandler
from ml.dataset.dataset_profiler import DatasetProfiler
from ml.dataset.dataset_statistics import DatasetStatistics
from ml.dataset.dataset_pipeline import DatasetPipeline, load_config
from verify_dataset_pipeline import verify_pipeline_integrity

# --- Test Data Fixtures ---

@pytest.fixture
def sample_raw_csv_data():
    return pd.DataFrame({
        "title": ["Fake News Title 1", "True News Title 2", "Empty title", ""],
        "text": ["Text of fake news article 1", "Text of true news article 2", "Text 3", "Text 4"],
        "subject": ["politics", "worldnews", "News", "politics"],
        "date": ["December 31, 2017", "December 30, 2017", "December 29, 2017", "December 28, 2017"]
    })

@pytest.fixture
def sample_raw_excel_data():
    return pd.DataFrame({
        "Statement": ["Indian Fake News Title 1", "Indian True News Title 2", "Indian Title 3"],
        "Category": ["Politics", "Health", "Business"],
        "Date": ["31-Dec-17", "30-Dec-17", "29-Dec-17"]
    })

# --- Unit Tests ---

def test_dataset_loader():
    loader = DatasetLoader()
    # Test loading unsupported extension
    with pytest.raises(ValueError, match="Unsupported file extension"):
        loader.load_dataset("fake_file.txt")


def test_dataset_validator(sample_raw_csv_data):
    validator = DatasetValidator()
    standardizer = DatasetStandardizer()
    
    # Test empty dataframe validation
    empty_df = pd.DataFrame()
    report = validator.validate_dataframe(
        empty_df, "empty", "ISOT"
    )
    assert not report["is_valid"]
    assert "empty" in report["errors"][0]

    # Test valid dataframe validation
    df_std = standardizer.standardize(
        sample_raw_csv_data, origin="ISOT", default_label=0
    )
    report = validator.validate_dataframe(
        df_std, "csv", "ISOT"
    )
    assert report["is_valid"]
    assert len(report["errors"]) == 0


def test_dataset_standardizer(sample_raw_csv_data):
    standardizer = DatasetStandardizer()
    
    # Standardize ISOT
    std_df = standardizer.standardize(
        sample_raw_csv_data, origin="ISOT", default_label=1
    )
    
    assert "id" in std_df.columns
    assert "dataset_origin" in std_df.columns
    assert (std_df["dataset_origin"] == "ISOT").all()
    assert (std_df["label"] == 1).all()
    assert std_df.loc[0, "published_date"] == "2017-12-31"
    assert std_df.loc[0, "category"] == "politics"
    assert std_df.loc[0, "language"] == "en"


def test_dataset_merger():
    merger = DatasetMerger()
    
    df1 = pd.DataFrame({
        "id": ["isot_1"], "title": ["T1"], "text": ["Tx1"], "label": [0],
        "source": [None], "category": ["Cat1"], "author": [None],
        "published_date": ["2017-12-31"], "language": ["en"], "url": [None],
        "dataset_origin": ["ISOT"]
    })
    
    df2 = pd.DataFrame({
        "id": ["india_1"], "title": ["T2"], "text": ["Tx2"], "label": [1],
        "source": [None], "category": ["Cat2"], "author": [None],
        "published_date": ["2017-12-30"], "language": ["en"], "url": [None],
        "dataset_origin": ["INDIA"]
    })

    merged_df = merger.merge([df1, df2])
    assert len(merged_df) == 2
    assert "id" in merged_df.columns
    assert merged_df.loc[0, "id"] == "isot_1"
    assert merged_df.loc[1, "id"] == "india_1"


def test_missing_value_handler():
    handler = MissingValueHandler()
    
    df = pd.DataFrame({
        "title": ["Title 1", None, "Title 3", "Title 4"],
        "text": ["Text 1", "Text 2", None, "Text 4"],
        "label": [0, 1, 0, None],
        "source": [None, "Source 2", "Source 3", "Source 4"]
    })

    cleaned_df, report = handler.handle_missing_values(df)
    
    # Row 0 is valid. Row 1 has missing title. Row 2 has missing text. Row 3 has missing label.
    # Therefore, only Row 0 should remain.
    assert len(cleaned_df) == 1
    assert cleaned_df.iloc[0]["title"] == "Title 1"
    assert report["total_dropped"] == 3
    assert len(report["removed_records"]) == 3
    assert "removal_reason" in report["removed_records"][0]


def test_duplicate_remover():
    remover = DuplicateRemover()
    
    df = pd.DataFrame({
        "id": ["master_1", "master_2", "master_3", "master_4"],
        "title": ["Unique Title", "Duplicate Title", "Duplicate Title", "Near Duplicate Title"],
        "text": ["Some unique text here.", "Identical text content.", "Identical text content.", "Identical text content. But with minor differences in the suffix."],
        "label": [0, 0, 1, 0],
        "source": [None, None, None, None],
        "category": [None, None, None, None],
        "published_date": [None, None, None, None],
        "dataset_origin": ["ISOT", "ISOT", "ISOT", "ISOT"]
    })

    cleaned_df, report = remover.remove_duplicates(df, similarity_threshold=0.85, batch_size=2)
    
    assert len(cleaned_df) < len(df)
    assert report["exact_text_duplicates_removed"] >= 1
    assert len(report["removed_records"]) >= 1
    assert "removal_reason" in report["removed_records"][0]


def test_dataset_profiler():
    profiler = DatasetProfiler()
    
    df = pd.DataFrame({
        "id": ["master_1", "master_2"],
        "title": ["Real News", "Fake News"],
        "text": ["Short news article 1 content.", "Very long news article 2 content with more words."],
        "label": [0, 1],
        "source": [None, "Reuters"],
        "category": ["politics", "world"],
        "author": [None, "John Doe"],
        "published_date": ["2017-12-31", "2017-12-30"],
        "language": ["en", "en"],
        "url": [None, None],
        "dataset_origin": ["ISOT", "INDIA"]
    })

    report = profiler.profile_dataset(df, duplicate_count=1)
    
    assert report["summary"]["total_articles"] == 2
    assert report["summary"]["real_articles"] == 1
    assert report["summary"]["fake_articles"] == 1
    assert report["summary"]["indian_articles"] == 1
    assert report["summary"]["isot_articles"] == 1
    assert report["text_statistics"]["average_word_length"] > 0
    assert report["missing_values"]["source"]["count"] == 1


def test_dataset_statistics():
    stats_gen = DatasetStatistics()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "dataset_statistics.json")
        stats = stats_gen.generate_statistics(
            raw_counts={"isot_true": 10, "isot_fake": 10},
            cleaned_counts={"isot_true": 9, "isot_fake": 8},
            final_count=15,
            duplicates_removed=2,
            missing_mandatory_dropped=0,
            real_count=8,
            fake_count=7,
            output_path=output_path
        )
        
        assert os.path.exists(output_path)
        assert stats["pipeline_summary"]["total_raw_rows_loaded"] == 20
        assert stats["pipeline_summary"]["total_final_rows_preserved"] == 15
        assert stats["class_distribution"]["real_news_count"] == 8


# --- New Production-Ready Task Tests ---

def test_configuration_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "dataset_config.yaml")
        # Test default fallback
        cfg = load_config(config_path)
        assert cfg["duplicate_detection"]["batch_size"] == 2000
        assert cfg["defaults"]["language"] == "en"

        # Test loading valid yaml
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("""
duplicate_detection:
  batch_size: 50
  similarity_threshold: 0.88
""")
        cfg = load_config(config_path)
        assert cfg["duplicate_detection"]["batch_size"] == 50
        assert cfg["duplicate_detection"]["similarity_threshold"] == 0.88
        assert cfg["defaults"]["language"] == "en" # Fallback key retained


def test_memory_safe_duplicate_detection_large_dataset():
    remover = DuplicateRemover()
    # Create a dummy large dataset (6 records)
    # We will set batch_size = 2, which triggers multiple batches
    df = pd.DataFrame({
        "id": [f"master_{i}" for i in range(1, 7)],
        "title": ["Title A", "Title A", "Title B", "Title C", "Title C", "Title D"],
        "text": ["Text content unique 1", "Text content unique 1", "Text content unique 2", "Text content unique 3", "Text content unique 3", "Text content unique 4"],
        "label": [0, 0, 1, 0, 1, 0],
        "source": [None]*6, "category": [None]*6, "published_date": [None]*6, "dataset_origin": ["ISOT"]*6
    })

    cleaned, report = remover.remove_duplicates(df, similarity_threshold=0.90, batch_size=2)
    # Duplicate records should be identified batch-wise
    assert len(cleaned) < 6
    assert report["exact_title_duplicates_removed"] == 2  # Title A (index 1) and Title C (index 4) are duplicates


def test_pipeline_recovery_on_dedup_failure():
    remover = DuplicateRemover()
    df = pd.DataFrame({
        "id": ["master_1"], "title": ["T1"], "text": ["Text unique"], "label": [0]
    })
    # Option 3 recovery fallback test: Mock Vectorizer to fail
    with patch("sklearn.feature_extraction.text.TfidfVectorizer.fit_transform", side_effect=MemoryError("Out of memory mock")):
        cleaned, report = remover.remove_duplicates(df, batch_size=2)
        # Should finish successfully without crash and report skip status
        assert report["semantic_deduplication_skipped"] is True
        assert len(cleaned) == 1


def test_integrity_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Before file creation, it should fail
        passed, errors = verify_pipeline_integrity(project_root=tmpdir)
        assert not passed
        assert len(errors) > 0

        # Create expected directory structure
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "raw", "isot"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "raw", "india"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "cleaned"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "merged"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "processed"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "statistics"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "ml", "dataset", "versions"), exist_ok=True)

        # Touch raw/cleaned/merged files
        for fpath in [
            "ml/dataset/raw/isot/Fake.csv", "ml/dataset/raw/isot/True.csv",
            "ml/dataset/raw/india/Indian_Fake_News_dataset.xlsx", "ml/dataset/raw/india/Indian_True_News_dataset.xlsx",
            "ml/dataset/cleaned/isot_true_cleaned.csv", "ml/dataset/cleaned/isot_fake_cleaned.csv",
            "ml/dataset/cleaned/india_true_cleaned.csv", "ml/dataset/cleaned/india_fake_cleaned.csv",
            "ml/dataset/merged/merged_dataset.csv",
        ]:
            with open(os.path.join(tmpdir, fpath), "w") as f: f.write("")

        # Create master csv
        df = pd.DataFrame({
            "id": ["master_1"], "title": ["Valid Title"], "text": ["Valid Text body"], "label": [0],
            "source": ["Reuters"], "category": ["world"], "author": ["John"],
            "published_date": ["2026-07-13"], "language": ["en"], "url": ["http://reuters.com"],
            "dataset_origin": ["ISOT"]
        })
        master_abs_path = os.path.join(tmpdir, "ml", "dataset", "processed", "master_dataset_v1.csv")
        df.to_csv(master_abs_path, index=False)

        # Write dummy stats & versions JSON
        with open(os.path.join(tmpdir, "ml", "dataset", "statistics", "dataset_statistics.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(tmpdir, "ml", "dataset", "statistics", "dataset_profile.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(tmpdir, "ml", "dataset", "versions", "dataset_versions.json"), "w") as f:
            json.dump([], f)

        # Write matching hash file
        sha = hashlib.sha256()
        with open(master_abs_path, "rb") as f:
            sha.update(f.read())
        with open(os.path.join(tmpdir, "ml", "dataset", "statistics", "dataset_hash.json"), "w") as f:
            json.dump({"SHA256": sha.hexdigest(), "File Name": "master_dataset_v1.csv"}, f)

        # Run integrity check again - should pass
        passed, errors = verify_pipeline_integrity(project_root=tmpdir)
        assert passed
        assert len(errors) == 0


# --- Integration Test with Mocked Loader ---

@patch("ml.dataset.dataset_loader.DatasetLoader.load_csv")
@patch("ml.dataset.dataset_loader.DatasetLoader.load_excel")
def test_dataset_pipeline(mock_load_excel, mock_load_csv):
    # Mock return values for loaders
    mock_load_csv.side_effect = [
        pd.DataFrame({
            "title": ["Real News 1", "Real News 2"],
            "text": ["Text of real news 1", "Text of real news 2"],
            "subject": ["politics", "politics"],
            "date": ["December 31, 2017", "December 30, 2017"]
        }),
        pd.DataFrame({
            "title": ["Fake News 1", "Fake News 2"],
            "text": ["Text of fake news 1", "Text of fake news 2"],
            "subject": ["worldnews", "worldnews"],
            "date": ["December 31, 2017", "December 30, 2017"]
        })
    ]
    
    mock_load_excel.side_effect = [
        pd.DataFrame({
            "Statement": ["Indian Real News 1"],
            "Category": ["Health"],
            "Date": ["31-Dec-17"],
            "text": ["Text of Indian real news 1"]
        }),
        pd.DataFrame({
            "Statement": ["Indian Fake News 1"],
            "Category": ["Politics"],
            "Date": ["30-Dec-17"],
            "text": ["Text of Indian fake news 1"]
        })
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock raw dataset folders
        isot_raw_dir = os.path.join(tmpdir, "ml", "dataset", "raw", "isot")
        india_raw_dir = os.path.join(tmpdir, "ml", "dataset", "raw", "india")
        os.makedirs(isot_raw_dir, exist_ok=True)
        os.makedirs(india_raw_dir, exist_ok=True)
        
        # Touch mock files
        with open(os.path.join(isot_raw_dir, "True.csv"), "w") as f: f.write("")
        with open(os.path.join(isot_raw_dir, "Fake.csv"), "w") as f: f.write("")
        with open(os.path.join(india_raw_dir, "Indian_True_News_dataset.xlsx"), "w") as f: f.write("")
        with open(os.path.join(india_raw_dir, "Indian_Fake_News_dataset.xlsx"), "w") as f: f.write("")

        # Create config file
        os.makedirs(os.path.join(tmpdir, "config"), exist_ok=True)
        config_path = os.path.join(tmpdir, "config", "dataset_config.yaml")
        with open(config_path, "w") as f:
            f.write("""
duplicate_detection:
  batch_size: 2
  similarity_threshold: 0.95
  max_features: 5000
schemas:
  supported_extensions: [".csv", ".xlsx", ".xls"]
  target_columns: ["id", "title", "text", "label", "source", "category", "author", "published_date", "language", "url", "dataset_origin"]
  mandatory_fields: ["title", "text", "label"]
defaults:
  language: "en"
  date_format: "%Y-%m-%d"
paths:
  raw_dir: "ml/dataset/raw"
  cleaned_dir: "ml/dataset/cleaned"
  merged_dir: "ml/dataset/merged"
  processed_dir: "ml/dataset/processed"
  statistics_dir: "ml/dataset/statistics"
  removed_dir: "ml/dataset/removed"
  versions_dir: "ml/dataset/versions"
  reports_dir: "ml/dataset/reports"
  logs_dir: "logs"
""")

        # Run pipeline
        pipeline = DatasetPipeline(project_root=tmpdir, config_file="config/dataset_config.yaml")
        success = pipeline.run()
        
        assert success
        
        # Verify output directories and files
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "cleaned", "isot_true_cleaned.csv"))
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "cleaned", "india_fake_cleaned.csv"))
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "merged", "merged_dataset.csv"))
        
        master_path = os.path.join(tmpdir, "ml", "dataset", "processed", "master_dataset_v1.csv")
        assert os.path.exists(master_path)
        
        # Verify removed archives
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "removed", "duplicate_rows.csv"))
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "removed", "missing_rows.csv"))
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "removed", "invalid_rows.csv"))

        # Verify quality report
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "reports", "dataset_quality_report.md"))

        # Verify hash and versions files
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "statistics", "dataset_hash.json"))
        assert os.path.exists(os.path.join(tmpdir, "ml", "dataset", "versions", "dataset_versions.json"))

        # Load output master dataset and check its contents
        master_df = pd.read_csv(master_path)
        assert len(master_df) == 6
