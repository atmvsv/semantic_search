from abc import ABC, abstractmethod

class BaseSearchEngine(ABC):
    @abstractmethod
    def index(self, corpus: dict[str, str]) -> None:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        ...