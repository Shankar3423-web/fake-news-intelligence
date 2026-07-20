import logging
from typing import Dict, Any

from ml.preprocessing.preprocessing_utils import get_shared_spacy_model

logger = logging.getLogger("verification_pipeline")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not installed in the current environment. SimilarityEngine will use word overlap cosine fallback.")

class SimilarityEngine:
    """
    SimilarityEngine compares an input article against retrieved evidence
    articles and calculates TF-IDF Cosine similarity, Jaccard index,
    and spaCy semantic similarity. Exposes fallback code if dependencies are missing.
    """
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        try:
            self.nlp = get_shared_spacy_model(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load shared spaCy model for SimilarityEngine: {e}")
            self.nlp = None

    def calculate_similarity(self, input_text: str, retrieved_text: str) -> Dict[str, float]:
        """
        Computes multiple similarity metrics and returns them along with a composite score.
        """
        if not input_text.strip() or not retrieved_text.strip():
            return {
                "cosine": 0.0,
                "jaccard": 0.0,
                "semantic": 0.0,
                "composite": 0.0
            }

        # 1. Cosine Similarity
        cosine_sim = 0.0
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
                tfidf = vectorizer.fit_transform([input_text, retrieved_text])
                cosine_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
            except Exception as e:
                logger.debug(f"TF-IDF similarity calculation failed: {e}")
                cosine_sim = self._calculate_word_overlap_cosine(input_text, retrieved_text)
        else:
            cosine_sim = self._calculate_word_overlap_cosine(input_text, retrieved_text)

        # 2. Jaccard Similarity
        jaccard_sim = 0.0
        try:
            tokens1 = set(input_text.lower().split())
            tokens2 = set(retrieved_text.lower().split())
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            if union:
                jaccard_sim = float(len(intersection) / len(union))
        except Exception as e:
            logger.debug(f"Jaccard similarity calculation failed: {e}")

        # 3. Semantic Similarity
        semantic_sim = 0.0
        if self.nlp:
            try:
                doc1 = self.nlp(input_text[:5000])  # limit text length for spaCy parser
                doc2 = self.nlp(retrieved_text[:5000])
                if doc1.vector_norm and doc2.vector_norm:
                    semantic_sim = float(doc1.similarity(doc2))
                else:
                    # Fallback if no vectors are available in small model
                    chunks1 = {c.text.lower() for c in doc1.noun_chunks}
                    chunks2 = {c.text.lower() for c in doc2.noun_chunks}
                    chunk_intersect = chunks1.intersection(chunks2)
                    chunk_union = chunks1.union(chunks2)
                    semantic_sim = float(len(chunk_intersect) / len(chunk_union)) if chunk_union else jaccard_sim
            except Exception as e:
                logger.debug(f"spaCy semantic similarity calculation failed: {e}")
                semantic_sim = jaccard_sim
        else:
            # Fallback to Jaccard as semantic proxy if spaCy is missing
            semantic_sim = jaccard_sim

        # Ensure scores are within [0.0, 1.0]
        cosine_sim = max(0.0, min(1.0, cosine_sim))
        jaccard_sim = max(0.0, min(1.0, jaccard_sim))
        semantic_sim = max(0.0, min(1.0, semantic_sim))

        # 4. Composite Score
        # Weighting: 50% Cosine, 30% Semantic, 20% Jaccard
        composite = (0.5 * cosine_sim) + (0.3 * semantic_sim) + (0.2 * jaccard_sim)
        
        return {
            "cosine": round(cosine_sim, 4),
            "jaccard": round(jaccard_sim, 4),
            "semantic": round(semantic_sim, 4),
            "composite": round(composite, 4)
        }

    def _calculate_word_overlap_cosine(self, text1: str, text2: str) -> float:
        """Fallback word-overlap cosine similarity when scikit-learn is missing."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        c1 = Counter(words1)
        c2 = Counter(words2)
        
        all_words = set(c1.keys()).union(set(c2.keys()))
        
        dot_product = sum(c1.get(w, 0) * c2.get(w, 0) for w in all_words)
        mag1 = sum(c1.get(w, 0) ** 2 for w in all_words) ** 0.5
        mag2 = sum(c2.get(w, 0) ** 2 for w in all_words) ** 0.5
        
        if mag1 * mag2 == 0:
            return 0.0
        return float(dot_product / (mag1 * mag2))
        
from collections import Counter
