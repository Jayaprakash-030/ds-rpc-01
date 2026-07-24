import ast
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.rag_config import RAGConfig

DEFAULT_API_URL = "http://localhost:8000/chat_test"
DEFAULT_MULTIHOP_DATA_PATH = REPO_ROOT / "evals" / "multihop_dataset.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "multihop_baseline_scores.csv"

def run_multihop_baseline(config: RAGConfig, api_url: str, dataset_path: Path, output_path: Path):
    # 1. Load the dataset
    if not Path(dataset_path).exists():
        print(f" Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)
    results = []

    print(f"Running Baseline | Model: {config.llm_model} | K: {config.top_k}")

    for i, row in df.iterrows():
        # Mapping your specific column names
        # user_input -> question
        # reference -> ground_truth
        # reference_contexts -> ground_truth_contexts
        question = row['user_input']
        gt_answer = row['reference']
        gt_contexts_raw = row.get("context", "")
        try:
            gt_contexts = ast.literal_eval(gt_contexts_raw) if isinstance(gt_contexts_raw, str) else gt_contexts_raw
        except (ValueError, SyntaxError):
            gt_contexts = []
        
        payload = {
            "message": question,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "db_path": f"./vector_db/db_{config.embedding_model.split('/')[-1]}_cs{config.chunk_size}",
            "hybrid_weight": config.hybrid_weight,
        }
        
        start_time = time.time()
        try:
            # Backend is mocked to 'admin' role in main.py
            response = requests.post(api_url, params=payload)
            response.raise_for_status()
            data = response.json()
            latency = (time.time() - start_time) * 1000
            
            bot_answer = data.get("answer", "")
            bot_sources = data.get("sources", []) # List of filenames
            
            results.append({
                "id": f"MH_{i}",
                "run_name": output_path.stem,
                "question": question,
                "ground_truth": gt_answer,
                "bot_answer": bot_answer,
                "ground_truth_context_preview": str(gt_contexts)[:200] + "...", # For easy viewing
                "retrieved_sources": bot_sources,
                "latency_ms": round(latency, 2),
                # --- Metrics for your Manual Audit ---
                "score_retrieval_quality": "", # 1-5: Did it find the right files?
                "score_llm_faithfulness": "", # 1-5: Did it hallucinate?
                "score_reasoning_quality": "", # 1-5: Did it connect the dots?
                "error_type": "" # e.g., "Missing File A", "Logic Error"
            })
            print(f" Processed MH_{i}")
            
        except Exception as e:
            print(f" Error on Multi-Hop {i}: {e}")

    # 3. Save to a dedicated Multi-Hop results file
    pd.DataFrame(results).to_csv(output_path, index=False)
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"Config saved to '{config_path}'.")
    
    # 4. Print Summary Stats
    total = len(results)
    print("\n" + "="*30)
    print(f" MULTI-HOP BASELINE COMPLETE")
    print(f"Total Questions: {total}")
    print(f"Results saved to: {output_path}")
    print("="*30)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--llm-model", default=RAGConfig.llm_model)
    parser.add_argument("--embedding-model", default=RAGConfig.embedding_model)
    parser.add_argument("--temperature", type=float, default=RAGConfig.temperature)
    parser.add_argument("--chunk-size", type=int, default=RAGConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RAGConfig.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=RAGConfig.top_k)
    parser.add_argument("--hybrid-weight", type=float, default=RAGConfig.hybrid_weight)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = RAGConfig(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        hybrid_weight=args.hybrid_weight,
    )
    output_path = DEFAULT_OUTPUT_PATH
    if args.run_name:
        output_path = output_path.with_name(f"{output_path.stem}_{args.run_name}{output_path.suffix}")

    run_multihop_baseline(
        config=cfg,
        api_url=args.api_url,
        dataset_path=DEFAULT_MULTIHOP_DATA_PATH,
        output_path=output_path,
    )
