import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.models.base import BaseSearchEngine
from src.preprocessing import TextPreprocessor


class SBERTEngine(BaseSearchEngine):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "mps") -> None:
        self.preprocessor = TextPreprocessor()
        self.model = SentenceTransformer(model_name, device=device)
        self.index_faiss: faiss.IndexFlatIP | None = None
        self.doc_ids: list[str] = []

    def index(self, corpus: dict[str, str]) -> None:
        self.doc_ids = list(corpus.keys())
        processed_docs = [self.preprocessor.clean_minimal(doc) for doc in corpus.values()]

        embeddings = self.model.encode(
            processed_docs,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index_faiss = faiss.IndexFlatIP(dim)
        self.index_faiss.add(embeddings)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.index_faiss is None:
            raise RuntimeError("FAISS index is not built. Call index() first.")

        processed_query = self.preprocessor.clean_minimal(query)
        query_embedding = self.model.encode(
            [processed_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index_faiss.search(query_embedding, top_k)

        return [
            (self.doc_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]