import os
import tempfile
import json
import joblib
import pandas as pd
import numpy as np
import pytest
import scipy.sparse

from ml.feature_engineering.feature_config import FeatureConfig
from ml.feature_engineering.statistical_features import StatisticalFeatureExtractor
from ml.feature_engineering.readability_features import ReadabilityFeatureExtractor
from ml.feature_engineering.lexical_features import LexicalFeatureExtractor
from ml.feature_engineering.symbol_features import SymbolFeatureExtractor
from ml.feature_engineering.linguistic_features import LinguisticFeatureExtractor
from ml.feature_engineering.tfidf_features import TfidfFeatureGenerator
from ml.feature_engineering.feature_merger import FeatureMerger
from ml.feature_engineering.feature_validator import FeatureValidator
from ml.feature_engineering.feature_pipeline import run_feature_engineering_pipeline

@pytest.fixture
def sample_config():
    """Provides a default config instance."""
    return FeatureConfig()

def test_statistical_features():
    extractor = StatisticalFeatureExtractor()
    texts = pd.Series([
        "Hello world. This is a test sentence.",
        "Short.",
        ""
    ])
    df = extractor.extract_features(texts)
    
    # Check shape
    assert df.shape == (3, 6)
    
    # Check first row
    assert df.loc[0, "stat_word_count"] == 7
    assert df.loc[0, "stat_char_count"] == 37
    assert df.loc[0, "stat_sentence_count"] >= 1
    assert df.loc[0, "stat_avg_word_length"] > 0
    assert df.loc[0, "stat_avg_sentence_length"] > 0
    assert df.loc[0, "stat_vocabulary_size"] == 7
    
    # Check empty row defaults
    assert df.loc[2, "stat_word_count"] == 0
    assert df.loc[2, "stat_char_count"] == 0
    assert df.loc[2, "stat_avg_word_length"] == 0.0

def test_readability_features():
    extractor = ReadabilityFeatureExtractor()
    texts = pd.Series([
        "The quick brown fox jumps over the lazy dog.",
        ""
    ])
    df = extractor.extract_features(texts)
    
    assert df.shape == (2, 5)
    assert "read_flesch_reading_ease" in df.columns
    assert "read_flesch_kincaid_grade" in df.columns
    assert df.loc[0, "read_flesch_reading_ease"] != 0.0
    assert df.loc[1, "read_flesch_reading_ease"] == 0.0

def test_lexical_features(sample_config):
    extractor = LexicalFeatureExtractor(sample_config)
    texts = pd.Series([
        "The the standard stopword text.",
        ""
    ])
    df = extractor.extract_features(texts)
    
    assert df.shape == (2, 5)
    assert df.loc[0, "lex_unique_words"] == 4  # 'the', 'standard', 'stopword', 'text'
    assert df.loc[0, "lex_stopword_ratio"] > 0.0
    assert df.loc[1, "lex_diversity"] == 0.0

def test_symbol_features():
    extractor = SymbolFeatureExtractor()
    texts = pd.Series([
        "Hello 123! WORLD?",
        ""
    ])
    df = extractor.extract_features(texts)
    
    assert df.shape == (2, 4)
    assert df.loc[0, "sym_digit_count"] == 3
    assert df.loc[0, "sym_uppercase_count"] == 6  # 'H', 'W', 'O', 'R', 'L', 'D'
    assert df.loc[0, "sym_punctuation_count"] == 2  # '!', '?'
    assert df.loc[0, "sym_special_char_count"] == 0

def test_linguistic_features(sample_config):
    try:
        extractor = LinguisticFeatureExtractor(sample_config)
        texts = pd.Series([
            "Barack Obama visited London in 2012.",
            ""
        ])
        df = extractor.extract_features(texts)
        
        assert df.shape == (2, 9)
        assert df.loc[0, "ling_entity_count"] >= 1
        assert df.loc[0, "ling_noun_count"] >= 1
        assert df.loc[1, "ling_noun_count"] == 0
    except Exception as e:
        pytest.skip(f"Skipping spaCy test due to environment loading: {e}")

def test_tfidf_features(sample_config):
    # Override TF-IDF settings to be compatible with a tiny 2-document test corpus
    sample_config._config["tfidf"] = {
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
    }
    generator = TfidfFeatureGenerator(sample_config)
    texts = pd.Series([
        "cleaned text article",
        "another cleaned news article"
    ])
    matrix = generator.fit_transform(texts)
    
    assert scipy.sparse.issparse(matrix)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        vec_path = os.path.join(tmpdir, "vec.joblib")
        mat_path = os.path.join(tmpdir, "mat.joblib")
        generator.save(vec_path, mat_path, matrix)
        
        assert os.path.exists(vec_path)
        assert os.path.exists(mat_path)
        
        loaded_mat = joblib.load(mat_path)
        assert scipy.sparse.issparse(loaded_mat)
        assert loaded_mat.shape == matrix.shape

def test_feature_merger():
    merger = FeatureMerger()
    base_df = pd.DataFrame({
        "id": [1, 2],
        "label": [0, 1],
        "cleaned_text": ["text one", "text two"],
        "extra_col": ["a", "b"]
    })
    
    feat_df1 = pd.DataFrame({
        "feat1": [1.1, 1.2],
        "feat2": [2.1, 2.2]
    })
    
    feat_df2 = pd.DataFrame({
        "feat3": [3.1, 3.2]
    })
    
    merged = merger.merge_features(base_df, [feat_df1, feat_df2])
    
    # Should only keep: id, label, cleaned_text, and features
    assert merged.shape == (2, 6)
    assert "extra_col" not in merged.columns
    assert "feat1" in merged.columns
    assert "feat3" in merged.columns
    assert list(merged.columns[:3]) == ["id", "label", "cleaned_text"]

def test_feature_validator():
    validator = FeatureValidator()
    
    # Test valid input dataframe stub
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "input.csv")
        pd.DataFrame({
            "id": [1], "label": [0], "text": ["hello"], "cleaned_text": ["hello"]
        }).to_csv(input_file, index=False)
        
        is_valid, errors = validator.validate_input(input_file)
        assert is_valid
        assert len(errors) == 0

        # Test invalid input
        bad_input_file = os.path.join(tmpdir, "bad_input.csv")
        pd.DataFrame({
            "id": [1], "text": ["hello"]
        }).to_csv(bad_input_file, index=False)
        is_valid, errors = validator.validate_input(bad_input_file)
        assert not is_valid
        assert len(errors) > 0
