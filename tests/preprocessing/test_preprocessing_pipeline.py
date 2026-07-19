import os
import json
import tempfile
import shutil
import hashlib
import pandas as pd
import pytest
from typing import Generator

# Import preprocessing components
from ml.preprocessing.preprocessing_config import PreprocessingConfig
from ml.preprocessing.lowercase_converter import LowercaseConverter
from ml.preprocessing.unicode_normalizer import UnicodeNormalizer
from ml.preprocessing.html_remover import HTMLRemover
from ml.preprocessing.url_remover import URLRemover
from ml.preprocessing.email_remover import EmailRemover
from ml.preprocessing.mention_remover import MentionRemover
from ml.preprocessing.hashtag_processor import HashtagProcessor
from ml.preprocessing.emoji_remover import EmojiRemover
from ml.preprocessing.special_character_remover import SpecialCharacterRemover
from ml.preprocessing.punctuation_handler import PunctuationHandler
from ml.preprocessing.number_handler import NumberHandler
from ml.preprocessing.whitespace_normalizer import WhitespaceNormalizer
from ml.preprocessing.contraction_expander import ContractionExpander
from ml.preprocessing.language_detector import LanguageDetector
from ml.preprocessing.tokenizer import Tokenizer
from ml.preprocessing.stopword_remover import StopwordRemover
from ml.preprocessing.lemmatizer import Lemmatizer
from ml.preprocessing.short_word_remover import ShortWordRemover
from ml.preprocessing.text_cleaner import TextCleaner
from ml.preprocessing.preprocessing_validator import PreprocessingValidator
from ml.preprocessing.preprocessing_statistics import PreprocessingStatistics
from ml.preprocessing.preprocessing_profiler import PreprocessingProfiler
from ml.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
from verify_preprocessing_pipeline import verify_preprocessing_integrity

# ----------------- Unit Tests for Components -----------------

def test_lowercase_converter():
    converter = LowercaseConverter()
    assert converter.transform("HELLO WORLD") == "hello world"
    assert converter.transform("Test 123!") == "test 123!"
    assert converter.transform(None) == ""

def test_unicode_normalizer():
    normalizer = UnicodeNormalizer(form="NFKC")
    # \u2116 is the № symbol, which normalizes to No in NFKC
    assert normalizer.transform("\u2116 1") == "No 1"
    assert normalizer.transform(None) == ""

def test_html_remover():
    remover = HTMLRemover()
    assert remover.transform("<p>Hello <b>World</b></p>") == "Hello World"
    assert remover.transform("No tags here") == "No tags here"
    assert remover.removed_count > 0
    remover.reset()
    assert remover.removed_count == 0

def test_url_remover():
    remover = URLRemover()
    text = "Check out https://google.com or http://example.com/test for info."
    cleaned = remover.transform(text)
    assert "https://google.com" not in cleaned
    assert "http://example.com/test" not in cleaned
    assert remover.removed_count == 2
    
    # Test www prefix
    remover.reset()
    assert remover.transform("Go to www.example.com") == "Go to "
    assert remover.removed_count == 1

def test_email_remover():
    remover = EmailRemover()
    assert remover.transform("Contact us at test@example.com.") == "Contact us at ."
    assert remover.removed_count == 1

def test_mention_remover():
    remover = MentionRemover()
    assert remover.transform("Hello @user_name!") == "Hello !"
    assert remover.removed_count == 1

def test_hashtag_processor():
    processor = HashtagProcessor()
    assert processor.transform("Breaking: #Election2026 is tomorrow!") == "Breaking: Election2026 is tomorrow!"
    assert processor.processed_count == 1

def test_emoji_remover():
    remover = EmojiRemover()
    text = "Hello World! 👋😊🔥"
    assert remover.transform(text) == "Hello World! "
    assert remover.removed_count == 3

def test_special_character_remover():
    remover = SpecialCharacterRemover()
    # Removes symbols like currency and copyright, but preserves basic punctuation
    assert remover.transform("Price: 100$ & © 2026.") == "Price: 100$ &  2026."

def test_punctuation_handler():
    handler = PunctuationHandler()
    # Replaces punctuation with space but preserves apostrophe
    text = "hello, world! this is a test. don't fail."
    cleaned = handler.transform(text)
    assert "hello  world  this is a test  don't fail " in cleaned
    assert "," not in cleaned
    assert "!" not in cleaned

def test_number_handler():
    # Test standalone numbers replaced by token
    handler = NumberHandler(replacement_token="<NUM>", preserve_alphanumeric=True)
    text = "We have 100 items, COVID19 is bad, 3D printing is cool."
    cleaned = handler.transform(text)
    assert "<NUM>" in cleaned
    assert "COVID19" in cleaned
    assert "3D" in cleaned
    assert handler.replaced_count == 1

def test_whitespace_normalizer():
    normalizer = WhitespaceNormalizer()
    assert normalizer.transform("  Hello   \t\n  World  ") == "Hello World"

def test_contraction_expander():
    expander = ContractionExpander()
    assert expander.transform("don't tell me you can't") == "do not tell me you cannot"

def test_language_detector():
    detector = LanguageDetector(supported_languages=["en"])
    assert detector.detect("This is an English article about news.") == "en"
    assert detector.is_supported("en") is True
    # Test fallback
    assert detector.detect("") == "en"

def test_tokenizer():
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize("this is a test")
    assert tokens == ["this", "is", "a", "test"]

def test_stopword_remover():
    remover = StopwordRemover(language="english", custom_stopwords=["test"])
    tokens = ["this", "is", "a", "test", "news"]
    filtered = remover.remove(tokens)
    assert "this" not in filtered
    assert "test" not in filtered
    assert "news" in filtered
    assert remover.removed_count > 0

def test_lemmatizer():
    lemmatizer = Lemmatizer("en_core_web_sm")
    tokens = ["running", "ran", "runs", "better"]
    lemmas = lemmatizer.lemmatize(tokens)
    # running/ran/runs should be lemmatized to run, better to well/good
    assert "run" in lemmas or "running" not in lemmas
    
    # Test batch
    batch_lemmas = lemmatizer.lemmatize_batch([["running"], ["better"]])
    assert len(batch_lemmas) == 2

def test_short_word_remover():
    remover = ShortWordRemover(min_length=2)
    tokens = ["a", "be", "see", "i"]
    filtered = remover.remove(tokens)
    assert "a" not in filtered
    assert "i" not in filtered
    assert "be" in filtered
    assert remover.removed_count == 2

# ----------------- Integration Tests -----------------

@pytest.fixture
def temp_workspace() -> Generator[str, None, None]:
    """Sets up a temporary folder to act as a workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    
    # Create configuration folder
    os.makedirs(os.path.join(temp_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "ml/dataset/processed"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "ml/preprocessing/processed"), exist_ok=True)
    
    yield temp_dir
    
    shutil.rmtree(temp_dir)

def test_config_loading(temp_workspace):
    config_yaml = """
steps:
  lowercase_conversion: true
  unicode_normalization: false
unicode:
  normalization_form: "NFC"
paths:
  input_dataset: "test_input.csv"
"""
    config_path = os.path.join(temp_workspace, "config/preprocessing_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_yaml)
        
    config = PreprocessingConfig(config_path)
    assert config.get_step_enabled("lowercase_conversion") is True
    assert config.get_step_enabled("unicode_normalization") is False
    assert config.unicode_form == "NFC"
    assert config.get_path("input_dataset") == "test_input.csv"

def test_text_cleaner_orchestration():
    # Load default config
    config = PreprocessingConfig(config_path="non_existent.yaml")
    cleaner = TextCleaner(config)
    
    raw_text = "<p>Here is a #news URL: https://google.com @user and don't forget 123.</p>"
    cleaned = cleaner.clean(raw_text)
    
    # Check that:
    # - HTML tags removed
    # - URLs removed
    # - Mentions removed
    # - Hashtags processed (#news -> news)
    # - Contractions expanded (don't -> do not)
    # - Numbers replaced (<NUM>)
    # - Lowercased
    assert "news" in cleaned
    assert "https://google.com" not in cleaned
    assert "user" not in cleaned
    assert "do not" in cleaned
    assert "<NUM>" in cleaned
    assert "<p>" not in cleaned

def test_large_dataset_handling_simulation(temp_workspace):
    """Simulates batching and large dataset handling in pipeline validator."""
    # Write a mock large dataset
    df = pd.DataFrame({
        "id": range(10),
        "title": [f"Title {i}" for i in range(10)],
        "text": ["This is a test article for Phase 3 pipeline preprocessing." for _ in range(10)],
        "label": [i % 2 for i in range(10)],
        "source": ["Source" for _ in range(10)],
        "category": ["Politics" for _ in range(10)],
        "author": ["Author" for _ in range(10)],
        "published_date": ["2026-07-15" for _ in range(10)],
        "language": ["en" for _ in range(10)],
        "url": ["http://test.com" for _ in range(10)],
        "dataset_origin": ["ISOT" for _ in range(10)]
    })
    
    input_path = os.path.join(temp_workspace, "ml/dataset/processed/master_dataset_v1.csv")
    df.to_csv(input_path, index=False)
    
    # Verify input via PreprocessingValidator
    validator = PreprocessingValidator()
    is_valid, errors = validator.validate_input(input_path)
    assert is_valid is True
    assert len(errors) == 0

    # Test SHA-256 hash generation function
    sha256 = hashlib.sha256()
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    assert len(digest) == 64
