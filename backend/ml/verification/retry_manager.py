import time
import logging
from typing import Callable, Any, TypeVar

logger = logging.getLogger("verification_pipeline")

T = TypeVar("T")

class RetryManager:
    """
    RetryManager wraps function execution with an exponential backoff retry mechanism.
    """
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0) -> None:
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Executes the provided function with arguments, retrying on failure.
        """
        delay = self.initial_delay
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")
                    raise e
                
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} of {func.__name__} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                time.sleep(delay)
                delay *= self.backoff_factor

        raise last_exception if last_exception else Exception("Retry failed")
