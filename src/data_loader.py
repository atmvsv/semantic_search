import json
from pathlib import Path
from typing import Final
from dataclasses import dataclass, asdict

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class IRDataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    relevant_docs: dict[str, list[str]]


class QQPDataLoader:
    DATASET_NAME: Final[str] = "glue"
    DATASET_CONFIG: Final[str] = "qqp"

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_and_prepare(self, num_pairs: int = 50000) -> IRDataset:
        dataset = load_dataset(self.DATASET_NAME, self.DATASET_CONFIG, split="train")
        df = dataset.to_pandas()

        duplicates_df = df[df["label"] == 1].dropna().copy()
        sampled_df = duplicates_df.sample(
            n=min(num_pairs, len(duplicates_df)), random_state=42
        )

        corpus: dict[str, str] = {}
        queries: dict[str, str] = {}
        relevant_docs: dict[str, list[str]] = {}

        for _, row in sampled_df.iterrows():
            q1_id = f"q1_{row['idx']}"
            q2_id = f"q2_{row['idx']}"

            q1_text = str(row["question1"]).strip()
            q2_text = str(row["question2"]).strip()

            if q1_text and q2_text:
                queries[q1_id] = q1_text
                corpus[q2_id] = q2_text
                relevant_docs[q1_id] = [q2_id]

        ir_dataset = IRDataset(corpus=corpus, queries=queries, relevant_docs=relevant_docs)
        self._save_dataset(ir_dataset)

        return ir_dataset

    def _save_dataset(self, dataset: IRDataset) -> None:
        output_path = self.processed_dir / "qqp_ir_dataset.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(dataset), f, ensure_ascii=False, indent=2)

    def load_local_dataset(self) -> IRDataset:
        input_path = self.processed_dir / "qqp_ir_dataset.json"
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset not found at {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return IRDataset(**data)