import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.experiment_config import RAGConfig

# --- Configuration ---
# BASE_DIR and REPO_ROOT defined above
DEFAULT_API_URL = "http://localhost:8000/chat_test"
DEFAULT_GOLDEN_DATA_PATH = REPO_ROOT / "evals" / "golden_dataset.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "quality_baseline_scores.csv"


def run_quality_baseline(config: RAGConfig, api_url: str, dataset_path: Path, output_path: Path):
    try:
        df = pd.read_csv(dataset_path)
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="latin1")
    results = []

    print(f"Starting Intelligence Baseline | Model: {config.llm_model} | K: {config.top_k}")

    for i, row in df.iterrows():
        payload = {
            "message": row["question"],
            "top_k": config.top_k,
            "temperature": config.temperature,
            "db_path": f"./vector_db/db_{config.embedding_model.split('/')[-1]}_cs{config.chunk_size}",
        }
        try:
            start_t = time.time()
            response = requests.post(api_url, params=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_t) * 1000
            results.append({
                "id": f"Q_{i}",
                "run_name": output_path.stem,
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "bot_answer": data.get("answer"),
                "retrieved_chunks": data.get("sources"),
                "latency_ms": round(latency_ms, 2),
                "score_retrieval_quality": "",
                "score_llm_faithfulness": "",
                "score_answer_relevance": "",
                "error_type": ""
            })
        except Exception as exc:
            print(f"Error on Q_{i}: {exc}")

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Quality Baseline complete! Open '{output_path}'.")
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"Config saved to '{config_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--llm-model", default=RAGConfig.llm_model)
    parser.add_argument("--embedding-model", default=RAGConfig.embedding_model)
    parser.add_argument("--temperature", type=float, default=RAGConfig.temperature)
    parser.add_argument("--chunk-size", type=int, default=RAGConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RAGConfig.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=RAGConfig.top_k)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = RAGConfig(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    output_path = DEFAULT_OUTPUT_PATH
    if args.run_name:
        output_path = output_path.with_name(f"{output_path.stem}_{args.run_name}{output_path.suffix}")

    run_quality_baseline(
        config=cfg,
        api_url=args.api_url,
        dataset_path=DEFAULT_GOLDEN_DATA_PATH,
        output_path=output_path,
    )
