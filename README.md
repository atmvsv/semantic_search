# Semantic Search Benchmark: Lexical vs. Neural Retrieval

A machine learning project to compare three information retrieval approaches: BM25, LSA, and Sentence-BERT. The goal is to evaluate the trade-offs between keyword-based matching and semantic vector search using the Quora Question Pairs (QQP) dataset.

## Key Results

* **Semantic Superiority:** SBERT achieved the highest Recall@10 (0.741), successfully retrieving duplicates with zero lexical overlap (e.g., *"How can I prevent eating disorders?"* vs. *"What suppresses eating disorder?"*).
* **SVD Limitations:** LSA (100 components) performed poorly on short text samples, confirming that linear projections fail to capture complex linguistic nuances compared to non-linear Transformer architectures.

## Tech Stack

*   **Python**
*   **Core libraries:** `sentence-transformers`, `scikit-learn`, `rank_bm25`, `FAISS`

## Project Structure

```
src/
├── models/      
├── data_loader.py  
├── preprocessing.py
└── evaluation.py
scripts/
└── experiment.py 
```

## How to Run

1.  **Clone the repository:**
```bash
git clone https://github.com/atmvsv/semantic_search.git
cd semantic_search
```
2. **Create and activate the environment:** Requires Conda or Mamba.
```bash
mamba env create -f environment.yml
mamba activate semantic-search
```

3. **Download the data:**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

4. **Run the script:**
```bash
export PYTHONPATH=$(pwd)
python scripts/experiment.py
```
## Benchmark Results

*Tested on 50,000 document corpus | 1,000 evaluation queries.*

| Model | MRR@10 | Recall@10 | Latency (ms/q) | Indexing Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **BM25** | 0.4873 | 0.6850 | 29.28 | **5.61** |
| **LSA** | 0.2457 | 0.3550 | **18.02** | 6.40 |
| **SBERT** | **0.5139** | **0.7410** | 23.72 | 27.01 |