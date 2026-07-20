import os
import logging
import time
from typing import List, Dict, Any
from ml.verification.verification_config import VerificationConfig
from ml.verification.newsapi_provider import NewsAPIProvider
from ml.verification.gnews_provider import GNewsProvider
from ml.verification.newsdata_provider import NewsDataProvider
from ml.verification.rate_limiter import RateLimiter
from ml.verification.retry_manager import RetryManager
from ml.verification.cache_manager import CacheManager

logger = logging.getLogger("verification_pipeline")

class ProviderManager:
    """
    ProviderManager coordinates executing enabled news providers, merging results,
    caching, rate limiting, and handling individual provider failures.
    """
    def __init__(
        self,
        config: VerificationConfig,
        cache_manager: CacheManager,
        rate_limiter: RateLimiter,
        retry_manager: RetryManager
    ) -> None:
        self.config = config
        self.cache_manager = cache_manager
        self.rate_limiter = rate_limiter
        self.retry_manager = retry_manager
        
        # Load API keys from environment
        news_api_key = os.getenv("NEWS_API_KEY", "")
        gnews_api_key = os.getenv("GNEWS_API_KEY", "")
        newsdata_api_key = os.getenv("NEWSDATA_API_KEY", "")
        
        self.providers = []
        timeout = config.timeout
        max_results = config.max_results_per_provider
        
        if config.enable_newsapi:
            self.providers.append(NewsAPIProvider(news_api_key, timeout, max_results))
        if config.enable_gnews:
            self.providers.append(GNewsProvider(gnews_api_key, timeout, max_results))
        if config.enable_newsdata:
            self.providers.append(NewsDataProvider(newsdata_api_key, timeout, max_results))

        logger.info(f"Loaded {len(self.providers)} news providers: {[p.name for p in self.providers]}")

    def fetch_all(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries all configured news providers with the search query concurrently.
        Handles provider failures independently and merges the results.
        """
        import concurrent.futures
        all_results = []
        
        def fetch_from_provider(provider):
            provider_name = provider.name
            
            # Check cache first
            cached_response = self.cache_manager.get(provider_name, query)
            if cached_response is not None:
                return cached_response
            
            # Rate limit before making outbound requests
            self.rate_limiter.wait(provider_name, self.config.rate_limit_delay)
            
            # Execute search with retry policy
            try:
                logger.info(f"Querying provider '{provider_name}'...")
                start_time = time.time()
                
                # Fetch news
                results = self.retry_manager.execute(provider.fetch_news, query)
                
                duration = time.time() - start_time
                logger.info(f"Provider '{provider_name}' finished in {duration:.2f}s and returned {len(results)} articles.")
                
                # Cache results
                self.cache_manager.set(provider_name, query, results)
                return results
            except Exception as e:
                logger.error(f"Provider '{provider_name}' failed to fetch news: {e}. Continuing pipeline...")
                return []

        if not self.providers:
            return all_results

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            futures = [executor.submit(fetch_from_provider, p) for p in self.providers]
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
                
        return all_results
