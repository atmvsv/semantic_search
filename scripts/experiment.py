import time
import json
import logging
from pathlib import Path
from typing import Any

from src.data_loader import QQPDataLoader
from src.models.bm25_model import BM25Engine
from src.models.lsa_model import LSAEngine
from src.models.sbert_model import SBERTEngine
from src.evaluation import IREvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    loader = QQPDataLoader()
    try:
        dataset = loader.load_local_dataset()
        logging.info("Loaded existing dataset.")
    except FileNotFoundError:
        logging.info("Dataset not found. Fetching and preparing...")
        dataset = loader.fetch_and_prepare(num_pairs=50000)

    evaluator = IREvaluator(dataset.relevant_docs)

    test_queries = dict(list(dataset.queries.items())[:1000])
    logging.info(f"Corpus size: {len(dataset.corpus)}. Test queries: {len(test_queries)}.")

    models = {
        "BM25": BM25Engine(),
        "LSA_100": LSAEngine(n_components=100),
        "SBERT": SBERTEngine(device="mps")
    }

    results: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        logging.info(f"--- Evaluating {model_name} ---")

        t0_index = time.perf_counter()
        model.index(dataset.corpus)
        indexing_time = time.perf_counter() - t0_index
        logging.info(f"{model_name} indexed in {indexing_time:.2f} seconds.")

        search_results = {}
        t0_search = time.perf_counter()

        for q_id, q_text in test_queries.items():
            search_results[q_id] = model.search(q_text, top_k=10)

        search_time = time.perf_counter() - t0_search
        avg_latency_ms = (search_time / len(test_queries)) * 1000
        logging.info(f"{model_name} search latency: {avg_latency_ms:.2f} ms/query.")

        metrics = evaluator.evaluate_system(search_results)

        results[model_name] = {
            "MRR@10": metrics["MRR"],
            "Recall@10": metrics["Recall"],
            "Indexing_Time_sec": indexing_time,
            "Latency_ms_per_query": avg_latency_ms
        }
        logging.info(f"{model_name} Metrics: {results[model_name]}")

    output_path = Path("./data/processed/experiment_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()