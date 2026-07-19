import logging
from collections import Counter
from typing import List, Dict, Any, Set

logger = logging.getLogger("verification_pipeline")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not installed in the current environment. DuplicateRemover will use word-overlap fallback.")

class DuplicateRemover:
    """
    DuplicateRemover filters out redundant news articles from merged provider results
    by matching exact URLs, titles, and analyzing text cosine similarity.
    """
    def __init__(self, duplicate_threshold: float = 0.85) -> None:
        self.duplicate_threshold = duplicate_threshold

    def remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes duplicate articles from the list based on URL, title, and content similarity.
        """
        if not articles:
            return []

        unique_articles: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()
        seen_titles: Set[str] = set()

        for art in articles:
            # 1. Direct URL check
            url = art.get("url", "").strip().lower()
            if url and url in seen_urls:
                logger.debug(f"Duplicate url skipped: {url}")
                continue

            # 2. Direct title check (alphanumeric only)
            title = art.get("title", "").strip().lower()
            simplified_title = "".join(char for char in title if char.isalnum())
            if simplified_title and simplified_title in seen_titles:
                logger.debug(f"Duplicate exact title skipped: '{art.get('title')}'")
                continue

            # 3. Textual similarity check against already registered unique articles
            is_duplicate = False
            for existing in unique_articles:
                # Compare title similarity
                title_sim = self._calculate_cosine_similarity(art.get("title", ""), existing.get("title", ""))
                if title_sim >= self.duplicate_threshold:
                    logger.debug(f"Duplicate title similarity ({title_sim:.2f} >= {self.duplicate_threshold}): '{art.get('title')}'")
                    is_duplicate = True
                    break

                # Compare content similarity
                content1 = art.get("content", "") or art.get("description", "")
                content2 = existing.get("content", "") or existing.get("description", "")
                if content1 and content2:
                    content_sim = self._calculate_cosine_similarity(content1, content2)
                    if content_sim >= self.duplicate_threshold:
                        logger.debug(f"Duplicate content similarity ({content_sim:.2f} >= {self.duplicate_threshold}): '{art.get('title')}'")
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_articles.append(art)
                if url:
                    seen_urls.add(url)
                if simplified_title:
                    seen_titles.add(simplified_title)

        logger.info(f"Duplicate remover reduced article count from {len(articles)} to {len(unique_articles)}")
        return unique_articles

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two strings with fallbacks."""
        if not text1.strip() or not text2.strip():
            return 0.0
            
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
                tfidf = vectorizer.fit_transform([text1, text2])
                sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                return float(sim)
            except Exception:
                pass
                
        # Custom word frequency overlap cosine similarity fallback
        try:
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
        except Exception:
            # Absolute fallback to Jaccard index
            w1 = set(text1.lower().split())
            w2 = set(text2.lower().split())
            if not w1 or not w2:
                return 0.0
            return float(len(w1 & w2) / len(w1 | w2))
