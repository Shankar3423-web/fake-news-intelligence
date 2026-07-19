import pandas as pd
import logging
import textstat
from typing import Dict, Any

logger = logging.getLogger("feature_engineering_pipeline")

class ReadabilityFeatureExtractor:
    """
    Extracts readability metrics from text.
    Features:
    - Flesch Reading Ease
    - Flesch-Kincaid Grade
    - SMOG
    - Gunning Fog
    - Coleman-Liau
    """
    def __init__(self) -> None:
        pass

    def extract_features(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extracts readability features from a series of raw texts.
        """
        logger.info("Extracting readability features...")
        features = []
        
        for text in texts:
            if pd.isna(text) or not isinstance(text, str) or not text.strip():
                features.append({
                    "read_flesch_reading_ease": 0.0,
                    "read_flesch_kincaid_grade": 0.0,
                    "read_smog": 0.0,
                    "read_gunning_fog": 0.0,
                    "read_coleman_liau": 0.0
                })
                continue
                
            try:
                # Flesch Reading Ease
                try:
                    flesch_ease = textstat.flesch_reading_ease(text)
                except Exception:
                    flesch_ease = 0.0
                    
                # Flesch-Kincaid Grade
                try:
                    flesch_kincaid = textstat.flesch_kincaid_grade(text)
                except Exception:
                    flesch_kincaid = 0.0
                    
                # SMOG
                try:
                    smog = textstat.smog_index(text)
                except Exception:
                    smog = 0.0
                    
                # Gunning Fog
                try:
                    gunning_fog = textstat.gunning_fog(text)
                except Exception:
                    gunning_fog = 0.0
                    
                # Coleman-Liau
                try:
                    coleman_liau = textstat.coleman_liau_index(text)
                except Exception:
                    coleman_liau = 0.0
                    
                features.append({
                    "read_flesch_reading_ease": float(round(flesch_ease, 4)),
                    "read_flesch_kincaid_grade": float(round(flesch_kincaid, 4)),
                    "read_smog": float(round(smog, 4)),
                    "read_gunning_fog": float(round(gunning_fog, 4)),
                    "read_coleman_liau": float(round(coleman_liau, 4))
                })
            except Exception as e:
                logger.error(f"Error extracting readability features for text: {e}")
                features.append({
                    "read_flesch_reading_ease": 0.0,
                    "read_flesch_kincaid_grade": 0.0,
                    "read_smog": 0.0,
                    "read_gunning_fog": 0.0,
                    "read_coleman_liau": 0.0
                })
                
        return pd.DataFrame(features)
