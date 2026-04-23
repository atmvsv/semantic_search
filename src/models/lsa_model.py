import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from src.models.base import BaseSearchEngine
from src.preprocessing import TextPreprocessor


class LSAEngine(BaseSearchEngine):
    def __init__(self, n_components: int = 100) -> None:
        self.preprocessor = TextPreprocessor()
        self.tfidf = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vectors: np.ndarray | None = None
        self.doc_ids: list[str] = []

    def index(self, corpus: dict[str, str]) -> None:
        self.doc_ids = list(corpus.keys())
        processed_docs = [" ".join(self.preprocessor.clean_full(doc)) for doc in corpus.values()]

        tfidf_matrix = self.tfidf.fit_transform(processed_docs)
        self.doc_vectors = self.svd.fit_transform(tfidf_matrix)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.doc_vectors is None:
            raise RuntimeError("Index is not built. Call index() first.")

        processed_query = " ".join(self.preprocessor.clean_full(query))
        query_tfidf = self.tfidf.transform([processed_query])
        query_svd = self.svd.transform(query_tfidf)

        similarities = cosine_similarity(query_svd, self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.doc_ids[i], float(similarities[i])) for i in top_indices]