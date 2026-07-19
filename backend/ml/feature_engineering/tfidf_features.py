import os
import joblib
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from ml.feature_engineering.feature_config import FeatureConfig

logger = logging.getLogger("feature_engineering_pipeline")

class TfidfFeatureGenerator:
    """
    Generates TF-IDF features from preprocessed text.
    Maintains a sparse matrix representation and saves artifacts using joblib.
    """
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.max_features = config.tfidf_max_features
        self.ngram_range = config.tfidf_ngram_range
        self.min_df = config.tfidf_min_df
        self.max_df = config.tfidf_max_df
        self.sublinear_tf = config.tfidf_sublinear_tf
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf
        )

    def fit_transform(self, cleaned_texts: pd.Series) -> spmatrix:
        """
        Fits the TF-IDF vectorizer and transforms the cleaned text.
        """
        logger.info(f"Fitting TF-IDF Vectorizer with max_features={self.max_features}...")
        # Handle any possible NaN values
        filled_texts = cleaned_texts.fillna("").astype(str)
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(filled_texts)
            logger.info(f"TF-IDF fit complete. Matrix shape: {tfidf_matrix.shape}")
            return tfidf_matrix
        except Exception as e:
            logger.critical(f"Failed to fit and transform TF-IDF: {e}")
            raise e

    def save(self, vectorizer_path: str, matrix_path: str, matrix: spmatrix) -> None:
        """
        Saves the fitted vectorizer and the sparse TF-IDF matrix using joblib.
        """
        # Ensure output directories exist
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        
        try:
            # Save Vectorizer
            logger.info(f"Saving TF-IDF vectorizer to {vectorizer_path}...")
            joblib.dump(self.vectorizer, vectorizer_path)
            logger.info("TF-IDF vectorizer saved successfully.")
            
            # Save Matrix
            logger.info(f"Saving TF-IDF sparse matrix to {matrix_path}...")
            joblib.dump(matrix, matrix_path)
            logger.info("TF-IDF sparse matrix saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save TF-IDF artifacts: {e}")
            raise e
