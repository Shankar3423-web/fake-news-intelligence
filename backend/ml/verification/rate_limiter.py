import time
import logging
from typing import Dict

logger = logging.getLogger("verification_pipeline")

class RateLimiter:
    """
    RateLimiter manages delays between requests to external APIs on a per-provider basis.
    """
    def __init__(self, default_delay: float = 1.0) -> None:
        self.default_delay = default_delay
        self.last_request_times: Dict[str, float] = {}

    def wait(self, provider_name: str, delay: float = None) -> None:
        """
        Blocks the current thread if the time elapsed since the last request
        to the provider is less than the required delay.
        """
        required_delay = delay if delay is not None else self.default_delay
        
        last_time = self.last_request_times.get(provider_name)
        if last_time is not None:
            elapsed = time.time() - last_time
            if elapsed < required_delay:
                sleep_duration = required_delay - elapsed
                logger.debug(f"Rate limiting provider '{provider_name}': sleeping for {sleep_duration:.2f}s")
                time.sleep(sleep_duration)
                
        self.last_request_times[provider_name] = time.time()
