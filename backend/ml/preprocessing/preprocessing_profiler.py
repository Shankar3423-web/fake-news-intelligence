import json
import os
import logging
from collections import Counter
from typing import Dict, Any, List

logger = logging.getLogger("preprocessing_pipeline")

class PreprocessingProfiler:
    """
    Analyzes and profiles the text vocabulary, lemma counts,
    and distributions of the preprocessed dataset.
    """
    
    def __init__(self, top_n: int = 100) -> None:
        self.top_n = top_n
        self.word_counter = Counter()
        self.lemma_counter = Counter()
        self.language_distribution: Dict[str, int] = {}
        self.total_articles = 0
        self.total_tokens = 0

    def update(self, words: List[str], lemmas: List[str], lang: str) -> None:
        """Updates profile metrics with words, lemmas, and language from an article."""
        self.total_articles += 1
        self.total_tokens += len(lemmas)
        
        self.word_counter.update(words)
        self.lemma_counter.update(lemmas)
        
        self.language_distribution[lang] = self.language_distribution.get(lang, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Compiles the profile into a structured dictionary."""
        avg_tokens = (self.total_tokens / self.total_articles) if self.total_articles > 0 else 0.0
        
        # Format top words/lemmas as lists of dicts or list of lists for clean JSON representation
        top_words_list = [{"word": w, "count": c} for w, c in self.word_counter.most_common(self.top_n)]
        top_lemmas_list = [{"lemma": l, "count": c} for l, c in self.lemma_counter.most_common(self.top_n)]
        
        return {
            "Top Words": top_words_list[:50],  # Limit to top 50 in root representation for readability
            "Top Lemmas": top_lemmas_list[:50],
            "Vocabulary Size": len(self.word_counter),
            "Unique Token Count": len(self.lemma_counter),
            "Most Frequent Terms": [item["lemma"] for item in top_lemmas_list[:10]],
            "Average Tokens Per Article": round(avg_tokens, 2),
            "Language Distribution": self.language_distribution
        }

    def save(self, file_path: str) -> None:
        """Saves the vocabulary profile to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Vocabulary profile successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save vocabulary profile to {file_path}: {e}")
