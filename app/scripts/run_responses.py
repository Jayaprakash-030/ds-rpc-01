import argparse
import ast
import json
import sys
import time
import re
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.experiment_config import RAGConfig

DEFAULT_API_URL = "http://localhost:8000/chat_test"


def coerce_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    return []


def compute_precision_recall(retrieved, reference):
    retrieved_set = {r for r in retrieved if r}
    reference_set = {r for r in reference if r}
    if not retrieved_set:
        precision = 0.0
    else:
        precision = len(retrieved_set & reference_set) / len(retrieved_set)
    if not reference_set:
        recall = 0.0
    else:
        recall = len(retrieved_set & reference_set) / len(reference_set)
    return precision, recall, sorted(retrieved_set & reference_set)


def normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]{4,}", text.lower())
    return set(tokens)


def context_match(retrieved: str, reference: str, min_overlap: int = 3) -> bool:
    r_tokens = normalize_tokens(retrieved)
    ref_tokens = normalize_tokens(reference)
    return len(r_tokens & ref_tokens) >= min_overlap


def compute_context_pr(retrieved_contexts, reference_contexts):
    retrieved = [str(c).strip() for c in retrieved_contexts if c]
    reference = [str(c).strip() for c in reference_contexts if c]
    if not retrieved:
        return 0.0, 0.0, []
    if not reference:
        return 0.0, 0.0, []

    matched_retrieved = []
    for r in retrieved:
        if any(context_match(r, ref) for ref in reference):
            matched_retrieved.append(r)

    precision = len(matched_retrieved) / len(retrieved) if retrieved else 0.0
    matched_reference = []
    for ref in reference:
        if any(context_match(r, ref) for r in retrieved):
            matched_reference.append(ref)
    recall = len(matched_reference) / len(reference) if reference else 0.0
    return precision, recall, matched_retrieved


def run_responses(config: RAGConfig, api_url: str, dataset_path: Path, output_path: Path):
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)
    results = []

    print(f"Running responses for {len(df)} questions...")

    for i, row in df.iterrows():
        question = row.get("question") or row.get("user_input")
        ground_truth = row.get("ground_truth") or row.get("reference")
        if not question:
            continue

        payload = {
            "message": question,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "db_path": f"./vector_db/db_{config.embedding_model.split('/')[-1]}_cs{config.chunk_size}",
            "use_hybrid": config.use_hybrid,
            "hybrid_weight": config.hybrid_weight,
        }

        try:
            start_t = time.time()
            response = requests.post(api_url, params=payload)
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.time() - start_t) * 1000
        except Exception as exc:
            print(f"Error on row {i}: {exc}")
            continue

        bot_answer = data.get("answer", "")
        retrieved_sources = data.get("sources", [])
        retrieved_contexts = data.get("retrieved_contexts", [])

        reference_sources = coerce_list(row.get("reference_sources", []))
        precision, recall, matched = compute_precision_recall(retrieved_sources, reference_sources)
        reference_contexts = coerce_list(row.get("context", []))
        ctx_precision, ctx_recall, ctx_matched = compute_context_pr(retrieved_contexts, reference_contexts)

        results.append({
            "id": f"Q_{i}",
            "question": question,
            "ground_truth": ground_truth,
            "bot_answer": bot_answer,
            "retrieved_sources": retrieved_sources,
            "reference_sources": reference_sources,
            "matched_sources": matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "context_precision": round(ctx_precision, 4),
            "context_recall": round(ctx_recall, 4),
            "latency_ms": round(latency_ms, 2),
            "score_retrieval_quality": "",
            "score_llm_faithfulness": "",
            "score_reasoning_quality": "",
            "error_type": "",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--llm-model", default=RAGConfig.llm_model)
    parser.add_argument("--embedding-model", default=RAGConfig.embedding_model)
    parser.add_argument("--temperature", type=float, default=RAGConfig.temperature)
    parser.add_argument("--chunk-size", type=int, default=RAGConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RAGConfig.chunk_overlap)
    parser.add_argument("--top-k", type=int, default=RAGConfig.top_k)
    parser.add_argument("--use-hybrid", action="store_true")
    parser.add_argument("--hybrid-weight", type=float, default=RAGConfig.hybrid_weight)
    args = parser.parse_args()

    cfg = RAGConfig(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        hybrid_weight=args.hybrid_weight,
    )

    if args.dataset_type in {"single", "both"}:
        dataset_path = REPO_ROOT / "evals" / "singlehop_dataset.csv"
        output_path = REPO_ROOT / "results" / f"{dataset_path.stem}_responses.csv"
        run_responses(cfg, args.api_url, dataset_path, output_path)

    if args.dataset_type in {"multi", "both"}:
        dataset_path = REPO_ROOT / "evals" / "multihop_dataset.csv"
        output_path = REPO_ROOT / "results" / f"{dataset_path.stem}_responses.csv"
        run_responses(cfg, args.api_url, dataset_path, output_path)
