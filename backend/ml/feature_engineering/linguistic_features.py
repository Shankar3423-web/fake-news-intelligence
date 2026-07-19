import pandas as pd
import logging
import spacy
from typing import Dict, Any, List
from ml.feature_engineering.feature_config import FeatureConfig

logger = logging.getLogger("feature_engineering_pipeline")

class LinguisticFeatureExtractor:
    """
    Extracts linguistic and syntactic features from raw text using spaCy.
    Features:
    - Named Entity Count
    - Noun Count (NOUN + PROPN)
    - Verb Count (VERB + AUX)
    - Adjective Count (ADJ)
    - POS Distribution (Noun, Verb, Adjective, Adverb, Pronoun ratios)
    """
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.model_name = config.spacy_model
        self.spacy_batch_size = config.spacy_batch_size
        self._nlp = self._load_spacy_model()

    def _load_spacy_model(self) -> spacy.language.Language:
        """Loads the spaCy model."""
        try:
            logger.info(f"Loading spaCy model '{self.model_name}'...")
            return spacy.load(self.model_name)
        except OSError:
            logger.warning(f"spaCy model '{self.model_name}' not found. Attempting to load 'en_core_web_sm' fallback...")
            try:
                return spacy.load("en_core_web_sm")
            except Exception as e:
                logger.critical(f"Failed to load any spaCy model: {e}")
                raise e

    def extract_features(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extracts linguistic features from a series of raw texts using spaCy batch processing.
        """
        logger.info(f"Extracting linguistic features with spaCy batch_size={self.spacy_batch_size}...")
        features = []
        
        # Replace NaN values with empty strings to avoid crashes in spaCy
        text_list = [str(t) if not pd.isna(t) else "" for t in texts]
        
        try:
            # Disable dependency parser and lemmatizer for speed and memory (tagger is needed for POS, ner for entities)
            docs = self._nlp.pipe(text_list, batch_size=self.spacy_batch_size, disable=["parser", "lemmatizer"])
            
            for doc in docs:
                total_tokens = len(doc)
                
                if total_tokens == 0:
                    features.append({
                        "ling_entity_count": 0,
                        "ling_noun_count": 0,
                        "ling_verb_count": 0,
                        "ling_adj_count": 0,
                        "ling_pos_noun_ratio": 0.0,
                        "ling_pos_verb_ratio": 0.0,
                        "ling_pos_adj_ratio": 0.0,
                        "ling_pos_adv_ratio": 0.0,
                        "ling_pos_pron_ratio": 0.0
                    })
                    continue
                
                noun_count = 0
                verb_count = 0
                adj_count = 0
                adv_count = 0
                pron_count = 0
                
                # Count POS tags
                for token in doc:
                    pos = token.pos_
                    if pos in ("NOUN", "PROPN"):
                        noun_count += 1
                    elif pos in ("VERB", "AUX"):
                        verb_count += 1
                    elif pos == "ADJ":
                        adj_count += 1
                    elif pos == "ADV":
                        adv_count += 1
                    elif pos == "PRON":
                        pron_count += 1
                
                entity_count = len(doc.ents)
                
                features.append({
                    "ling_entity_count": entity_count,
                    "ling_noun_count": noun_count,
                    "ling_verb_count": verb_count,
                    "ling_adj_count": adj_count,
                    "ling_pos_noun_ratio": float(round(noun_count / total_tokens, 4)),
                    "ling_pos_verb_ratio": float(round(verb_count / total_tokens, 4)),
                    "ling_pos_adj_ratio": float(round(adj_count / total_tokens, 4)),
                    "ling_pos_adv_ratio": float(round(adv_count / total_tokens, 4)),
                    "ling_pos_pron_ratio": float(round(pron_count / total_tokens, 4))
                })
        except Exception as e:
            logger.error(f"Error during spaCy linguistic feature extraction: {e}", exc_info=True)
            # Fill with fallback for the entire batch if a catastrophic failure occurs
            for _ in text_list:
                features.append({
                    "ling_entity_count": 0,
                    "ling_noun_count": 0,
                    "ling_verb_count": 0,
                    "ling_adj_count": 0,
                    "ling_pos_noun_ratio": 0.0,
                    "ling_pos_verb_ratio": 0.0,
                    "ling_pos_adj_ratio": 0.0,
                    "ling_pos_adv_ratio": 0.0,
                    "ling_pos_pron_ratio": 0.0
                })
                
        # Clean up references to free memory immediately
        if 'docs' in locals():
            del docs
        if 'text_list' in locals():
            del text_list
        import gc
        gc.collect()
        
        return pd.DataFrame(features)
