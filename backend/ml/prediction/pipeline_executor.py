import os
import logging
import joblib
import pandas as pd
import scipy.sparse
from typing import Dict, Any, List, Tuple

# Phase 3 Preprocessing Imports
from ml.preprocessing.preprocessing_config import PreprocessingConfig
from ml.preprocessing.text_cleaner import TextCleaner
from ml.preprocessing.language_detector import LanguageDetector
from ml.preprocessing.tokenizer import Tokenizer
from ml.preprocessing.stopword_remover import StopwordRemover
from ml.preprocessing.lemmatizer import Lemmatizer
from ml.preprocessing.short_word_remover import ShortWordRemover
from ml.preprocessing.preprocessing_utils import ensure_nltk_resources, ensure_spacy_model

# Phase 4 Feature Engineering Imports
from ml.feature_engineering.feature_config import FeatureConfig
from ml.feature_engineering.statistical_features import StatisticalFeatureExtractor
from ml.feature_engineering.readability_features import ReadabilityFeatureExtractor
from ml.feature_engineering.lexical_features import LexicalFeatureExtractor
from ml.feature_engineering.symbol_features import SymbolFeatureExtractor
from ml.feature_engineering.linguistic_features import LinguisticFeatureExtractor

logger = logging.getLogger("prediction_pipeline")

class PipelineExecutor:
    """
    Executes the exact preprocessing, feature engineering, and feature selection pipelines
    to convert raw text input into a standardized feature vector matching the model's training schema.
    """
    def __init__(
        self,
        preprocessing_config_path: str = "config/preprocessing_config.yaml",
        feature_config_path: str = "config/feature_config.yaml",
        tfidf_vectorizer_path: str = "ml/feature_engineering/processed/tfidf_vectorizer.joblib"
    ) -> None:
        self.preprocessing_config_path = preprocessing_config_path
        self.feature_config_path = feature_config_path
        self.tfidf_vectorizer_path = tfidf_vectorizer_path
        
        self._init_preprocessing()
        self._init_feature_engineering()
        self._load_tfidf_vectorizer()

    def _init_preprocessing(self) -> None:
        """Initializes all NLP preprocessing components from Phase 3."""
        logger.info("Initializing Phase 3 Preprocessing Pipeline components...")
        self.p3_config = PreprocessingConfig(self.preprocessing_config_path)
        
        # Ensure NLP resources are ready
        ensure_nltk_resources()
        spacy_ready = ensure_spacy_model(self.p3_config.spacy_model)
        if not spacy_ready:
            raise RuntimeError(f"Required spaCy model '{self.p3_config.spacy_model}' could not be loaded/installed.")
            
        self.text_cleaner = TextCleaner(self.p3_config)
        self.language_detector = LanguageDetector(
            supported_languages=self.p3_config.supported_languages,
            default_language=self.p3_config.default_language,
            fallback_on_error=self.p3_config.fallback_on_error
        )
        self.tokenizer = Tokenizer()
        self.stopword_remover = StopwordRemover(
            language=self.p3_config.stopword_language,
            custom_stopwords=self.p3_config.custom_stopwords
        )
        self.lemmatizer = Lemmatizer(self.p3_config.spacy_model)
        self.short_word_remover = ShortWordRemover(self.p3_config.min_token_length)
        logger.info("Phase 3 components initialized successfully.")

    def _init_feature_engineering(self) -> None:
        """Initializes all dense feature extractors from Phase 4."""
        logger.info("Initializing Phase 4 Feature Engineering extractors...")
        self.p4_config = FeatureConfig(self.feature_config_path)
        
        self.stat_extractor = StatisticalFeatureExtractor()
        self.read_extractor = ReadabilityFeatureExtractor()
        self.lex_extractor = LexicalFeatureExtractor(self.p4_config)
        self.sym_extractor = SymbolFeatureExtractor()
        
        self.ling_extractor = None
        if self.p4_config.get_step_enabled("linguistic_features"):
            self.ling_extractor = LinguisticFeatureExtractor(self.p4_config)
        logger.info("Phase 4 extractors initialized successfully.")

    def _load_tfidf_vectorizer(self) -> None:
        """Loads the fitted TF-IDF vectorizer joblib object."""
        logger.info(f"Loading TF-IDF vectorizer from {self.tfidf_vectorizer_path}...")
        if not os.path.exists(self.tfidf_vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found: {self.tfidf_vectorizer_path}")
        try:
            self.tfidf_vectorizer = joblib.load(self.tfidf_vectorizer_path)
        except Exception as e:
            raise ValueError(f"Failed to load TF-IDF vectorizer joblib: {e}")
        logger.info("TF-IDF vectorizer loaded successfully.")

    def preprocess_text(self, raw_text: str) -> Tuple[str, str]:
        """
        Executes Phase 3 preprocessing pipeline on a single raw text string.
        
        Returns:
            Tuple: (cleaned_text, language)
        """
        logger.info("Running text preprocessing...")
        
        # 1. Text cleaning (steps 2 to 14)
        cleaned_str = self.text_cleaner.clean(raw_text)
        
        # 2. Language detection
        lang = self.language_detector.detect(cleaned_str)
        if not self.language_detector.is_supported(lang):
            raise ValueError(f"Language '{lang}' is not supported by the preprocessing pipeline.")
            
        # 3. Tokenization
        tokens = self.tokenizer.tokenize(cleaned_str)
        
        # 4. Stopword removal
        tokens = self.stopword_remover.remove(tokens)
        
        # 5. Lemmatization
        lemmas = self.lemmatizer.lemmatize(tokens)
        
        # 6. Short word removal
        lemmas = self.short_word_remover.remove(lemmas)
        
        # 7. Reconstruct cleaned text
        final_clean_text = " ".join(lemmas)
        if not final_clean_text.strip():
            raise ValueError("Preprocessed text is empty after stopword and short word removal.")
            
        logger.info("Text preprocessing complete.")
        return final_clean_text, lang

    def extract_dense_features(self, raw_text: str) -> pd.DataFrame:
        """
        Extracts all enabled dense engineered features (Phase 4).
        """
        logger.info("Extracting dense features from raw text...")
        texts_series = pd.Series([raw_text])
        feature_dfs = []
        
        # Statistical features
        if self.p4_config.get_step_enabled("statistical_features"):
            stat_df = self.stat_extractor.extract_features(texts_series)
            feature_dfs.append(stat_df)
            
        # Readability features
        if self.p4_config.get_step_enabled("readability_features"):
            read_df = self.read_extractor.extract_features(texts_series)
            feature_dfs.append(read_df)
            
        # Lexical features
        if self.p4_config.get_step_enabled("lexical_features"):
            lex_df = self.lex_extractor.extract_features(texts_series)
            feature_dfs.append(lex_df)
            
        # Symbol features
        if self.p4_config.get_step_enabled("symbol_features"):
            sym_df = self.sym_extractor.extract_features(texts_series)
            feature_dfs.append(sym_df)
            
        # Linguistic features
        if self.p4_config.get_step_enabled("linguistic_features") and self.ling_extractor:
            ling_df = self.ling_extractor.extract_features(texts_series)
            feature_dfs.append(ling_df)
            
        # Concatenate column-wise
        if not feature_dfs:
            raise ValueError("No dense feature extraction step is enabled in feature_config.")
            
        df_dense = pd.concat(feature_dfs, axis=1)
        logger.info(f"Dense feature extraction complete. Shape: {df_dense.shape}")
        return df_dense

    def execute(self, raw_text: str, expected_feature_order: List[str]) -> Tuple[pd.DataFrame, str]:
        """
        Orchestrates full preprocessing, feature extraction, TF-IDF mapping, 
        and validates feature vector compatibility.
        
        Args:
            raw_text: Raw news article text
            expected_feature_order: Exact list of feature names expected by the model
            
        Returns:
            Tuple of:
                - feature_vector: DataFrame with 1 row, columns aligned with expected_feature_order
                - cleaned_text: Preprocessed text string
        """
        # Step 1: Preprocess text
        cleaned_text, lang = self.preprocess_text(raw_text)
        
        # Step 2: Extract dense features
        df_dense = self.extract_dense_features(raw_text)
        
        # Step 3: Generate TF-IDF features
        logger.info("Generating TF-IDF feature representation...")
        tfidf_matrix = self.tfidf_vectorizer.transform(pd.Series([cleaned_text]))
        
        # Step 4: Map & align to selected features
        logger.info("Aligning and building final feature vector...")
        features_dict = {}
        vocab = self.tfidf_vectorizer.vocabulary_
        
        for f in expected_feature_order:
            if f.startswith("tfidf_"):
                term = f[len("tfidf_"):]
                col_idx = vocab.get(term)
                if col_idx is not None:
                    # Retrieve coordinate from sparse row 0
                    features_dict[f] = float(tfidf_matrix[0, col_idx])
                else:
                    features_dict[f] = 0.0
            else:
                if f in df_dense.columns:
                    features_dict[f] = float(df_dense.iloc[0][f])
                else:
                    raise ValueError(f"Model expects dense feature '{f}' which was not engineered.")
                    
        # Construct final ordered DataFrame
        feature_vector = pd.DataFrame([features_dict], columns=expected_feature_order)
        
        # Step 5: Validate feature vector compatibility
        self.validate_feature_vector(feature_vector, expected_feature_order)
        
        return feature_vector, cleaned_text

    def validate_feature_vector(self, feature_vector: pd.DataFrame, expected_feature_order: List[str]) -> None:
        """Validates feature count, ordering, and names."""
        logger.info("Validating feature vector integrity...")
        
        # 1. Feature count check
        if len(feature_vector.columns) != len(expected_feature_order):
            raise ValueError(
                f"Feature count mismatch: Vector has {len(feature_vector.columns)} columns, "
                f"but model expects {len(expected_feature_order)}."
            )
            
        # 2. Feature names check
        missing_names = set(expected_feature_order) - set(feature_vector.columns)
        if missing_names:
            raise ValueError(f"Feature vector is missing required features: {missing_names}")
            
        # 3. Feature ordering check
        for idx, (col_name, expected_name) in enumerate(zip(feature_vector.columns, expected_feature_order)):
            if col_name != expected_name:
                raise ValueError(
                    f"Feature ordering mismatch at column index {idx}: "
                    f"Vector has '{col_name}', expected '{expected_name}'."
                )
                
        logger.info("Feature vector integrity check passed successfully.")
