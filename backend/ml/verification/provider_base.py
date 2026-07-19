from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseNewsProvider(ABC):
    """
    Abstract base class for all live news API providers.
    """
    def __init__(self, api_key: str, timeout: int = 10, max_results: int = 5) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.max_results = max_results

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique string name of the provider."""
        pass

    @abstractmethod
    def fetch_news(self, query: str) -> List[Dict[str, Any]]:
        """
        Queries the API, validates and normalizes the results,
        returning a list of standardized dictionaries representing articles.
        """
        pass
