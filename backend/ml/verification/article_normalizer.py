import logging
import re
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("verification_pipeline")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 is not installed in the current environment. ArticleNormalizer will use regex fallback.")

class ArticleNormalizer:
    """
    ArticleNormalizer sanitizes, formats, and standardizes news articles 
    retrieved from different API formats into a single schema.
    """
    def normalize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes and normalizes an article dictionary.
        """
        # Clean HTML out of textual fields
        title = self._clean_text(article.get("title", ""))
        description = self._clean_text(article.get("description", ""))
        content = self._clean_text(article.get("content", ""))
        
        # Normalize date to ISO format if possible
        published_date = article.get("published_date", "")
        normalized_date = self._normalize_date(published_date)

        # Standardize other fields
        language = str(article.get("language", "en")).lower()
        provider = str(article.get("provider", "unknown")).lower()
        source = str(article.get("source", "Unknown Source"))
        author = str(article.get("author", "Unknown"))
        url = str(article.get("url", ""))

        return {
            "title": title,
            "description": description,
            "content": content,
            "url": url,
            "source": source,
            "author": author,
            "published_date": normalized_date,
            "language": language,
            "provider": provider
        }

    def normalize_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalizes a list of articles."""
        return [self.normalize_article(art) for art in articles]

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
            
        # Strip HTML tags
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(text, "html.parser")
                cleaned = soup.get_text()
            except Exception:
                cleaned = re.sub(r'<[^>]*>', '', text)
        else:
            cleaned = re.sub(r'<[^>]*>', '', text)
        
        # Clean extra whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _normalize_date(self, date_str: str) -> str:
        if not date_str:
            return datetime.now().isoformat()
        
        # Strip trailing timezone info or clean format if needed
        # Try parsing various common formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%b %d, %Y",
            "%Y-%m-%d %H:%M:%S.%f"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue
                
        # If parsing fails, just return original string
        return date_str
