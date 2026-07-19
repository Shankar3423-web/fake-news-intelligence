import logging
from typing import List, Dict, Any, Set
from datetime import datetime, timezone
from ml.verification.similarity_engine import SimilarityEngine

logger = logging.getLogger("verification_pipeline")

# Predefined trusted sources for the source quality score boost
TRUSTED_SOURCES: Set[str] = {
    # Major wire services
    "reuters", "ap", "associated press", "afp", "agence france-presse", "united press international", "upi",
    # Major US/UK news organizations
    "bbc", "bbc news", "nytimes", "new york times", "the new york times",
    "bloomberg", "cnn", "the guardian", "guardian", "washington post", "the washington post",
    "wsj", "wall street journal", "the wall street journal",
    "abc news", "cbs news", "nbc news", "msnbc", "fox news", "pbs", "npr",
    "usa today", "los angeles times", "chicago tribune", "time", "newsweek",
    "the hill", "politico", "axios", "the atlantic",
    # International
    "al jazeera", "france 24", "dw", "deutsche welle", "sky news", "itv news",
    "times of india", "the times of india", "ndtv", "hindustan times", "indian express",
    "the hindu", "economic times", "mint", "livemint",
    "south china morning post", "nhk", "abc australia",
    # Science / Tech / Space
    "nasa", "space.com", "space daily", "spacedaily", "scientific american",
    "nature", "science", "new scientist", "phys.org", "ars technica", "wired",
    "the verge", "techcrunch", "engadget", "mit technology review",
    # Business / Finance
    "cnbc", "financial times", "ft", "forbes", "business insider", "fortune", "marketwatch",
    # Aggregators (legitimate)
    "yahoo", "yahoo news", "google news", "msn", "msn news",
    # Mock sources (for testing)
    "mock associated press", "mock reuters", "mock science daily", "mock gazette",
    "mock_tribune", "mock_chronicle"
}

class ArticleRanker:
    """
    ArticleRanker scores and ranks normalized articles based on similarity to input,
    source trust, recency, and keyword overlap.
    """
    def __init__(self, similarity_engine: SimilarityEngine) -> None:
        self.similarity_engine = similarity_engine

    def rank_articles(self, input_text: str, input_title: str, articles: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Calculates a ranking score for each article and returns the list sorted descending by score.
        """
        if not articles:
            return []

        ranked_articles = []
        keyword_set = {kw.lower() for kw in keywords}

        for art in articles:
            # 1. Similarity Scores
            title_sims = self.similarity_engine.calculate_similarity(input_title, art.get("title", ""))
            content_sims = self.similarity_engine.calculate_similarity(input_text, art.get("content", "") or art.get("description", ""))
            
            title_score = title_sims["composite"]
            content_score = content_sims["composite"]

            # 2. Keyword Overlap (Jaccard-like overlap with title and description)
            art_text = f"{art.get('title', '')} {art.get('description', '')}".lower()
            overlap_count = sum(1 for kw in keyword_set if kw in art_text)
            keyword_score = overlap_count / len(keyword_set) if keyword_set else 0.0

            # 3. Source Quality
            source_name = art.get("source", "").lower()
            is_trusted = False
            for trusted in TRUSTED_SOURCES:
                if trusted in source_name:
                    is_trusted = True
                    break
            source_score = 1.0 if is_trusted else 0.0

            # 4. Recency
            recency_score = self._calculate_recency_score(art.get("published_date", ""))

            # Calculate overall ranking score (weighted average)
            # Weights: 40% Content Similarity, 30% Title Similarity, 15% Keyword Overlap, 10% Recency, 5% Source Quality
            ranking_score = (
                (0.40 * content_score) +
                (0.30 * title_score) +
                (0.15 * keyword_score) +
                (0.10 * recency_score) +
                (0.05 * source_score)
            )

            # Store computed metrics back in the article dictionary
            art_with_score = art.copy()
            art_with_score["similarity_scores"] = {
                "title_similarity": title_score,
                "content_similarity": content_score,
                "composite": (title_score + content_score) / 2.0
            }
            art_with_score["ranking_score"] = round(ranking_score, 4)
            art_with_score["is_trusted_source"] = is_trusted
            
            ranked_articles.append(art_with_score)

        # Sort articles by ranking score descending
        ranked_articles.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        logger.info(f"Ranked {len(ranked_articles)} articles. Top score: {ranked_articles[0]['ranking_score'] if ranked_articles else 0.0}")
        return ranked_articles

    def _calculate_recency_score(self, date_str: str) -> float:
        """Computes a recency score in [0.0, 1.0] where 1.0 is extremely recent."""
        if not date_str:
            return 0.5
            
        try:
            # Parse ISO date format
            published_dt = datetime.fromisoformat(date_str)
            # Make timezone aware if it is not
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)
                
            now = datetime.now(timezone.utc)
            delta = now - published_dt
            
            hours = delta.total_seconds() / 3600.0
            if hours < 0:
                hours = 0
                
            # Time decay formula: score decays towards 0 as hours increase
            # 1.0 at 0 hours, ~0.5 at 48 hours, ~0.2 at a week
            recency = 1.0 / (1.0 + (hours / 48.0))
            return float(recency)
        except Exception as e:
            logger.debug(f"Failed to calculate recency score for date '{date_str}': {e}")
            return 0.5
