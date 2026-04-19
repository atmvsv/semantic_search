from typing import Dict, List, Tuple


class IREvaluator:
    def __init__(self, ground_truth: Dict[str, List[str]]):
        self.ground_truth = ground_truth

    def evaluate_mrr(self, query_id: str, retrieved_docs: List[Tuple[str, float]]) -> float:
        if query_id not in self.ground_truth:
            return 0.0

        relevant_docs = set(self.ground_truth[query_id])

        for rank, (doc_id, _) in enumerate(retrieved_docs, start=1):
            if doc_id in relevant_docs:
                return 1.0 / rank

        return 0.0

    def evaluate_recall(self, query_id: str, retrieved_docs: List[Tuple[str, float]]) -> float:
        if query_id not in self.ground_truth:
            return 0.0

        relevant_docs = set(self.ground_truth[query_id])
        if not relevant_docs:
            return 0.0

        retrieved_set = set([doc_id for doc_id, _ in retrieved_docs])
        found_relevant = len(relevant_docs.intersection(retrieved_set))

        return float(found_relevant) / len(relevant_docs)

    def evaluate_system(self, search_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, float]:
        total_mrr = 0.0
        total_recall = 0.0
        num_queries = len(search_results)

        if num_queries == 0:
            return {"MRR": 0.0, "Recall": 0.0}

        for query_id, retrieved_docs in search_results.items():
            total_mrr += self.evaluate_mrr(query_id, retrieved_docs)
            total_recall += self.evaluate_recall(query_id, retrieved_docs)

        return {
            "MRR": total_mrr / num_queries,
            "Recall": total_recall / num_queries
        }