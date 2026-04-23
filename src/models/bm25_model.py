import numpy as np
from rank_bm25 import BM25Okapi

from src.models.base import BaseSearchEngine
from src.preprocessing import TextPreprocessor


class BM25Engine(BaseSearchEngine):
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []

    def index(self, corpus: dict[str, str]) -> None:
        self.doc_ids = list(corpus.keys())
        tokenized_corpus = [self.preprocessor.clean_full(doc) for doc in corpus.values()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("Index is not built. Call index() first.")

        tokenized_query = self.preprocessor.clean_full(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]