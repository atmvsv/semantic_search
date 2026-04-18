from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class BaseSearchEngine(ABC):
    @abstractmethod
    def index(self, corpus: Dict[str, str]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        ...