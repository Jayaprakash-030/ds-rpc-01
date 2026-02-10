import argparse
import ast
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.evaluation import EvaluationDataset, LangchainLLMWrapper, SingleTurnSample, evaluate
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.experiment_config import RAGConfig

# Load .env relative to repo root so scripts work from any CWD
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

DEFAULT_API_URL = "http://localhost:8001/chat_test"


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

def normalize_sources(sources):
    normalized = []
    for src in sources:
        if not src:
            continue
        text = str(src).strip()
        # Keep filenames consistent when absolute paths are present
        if "/" in text or "\\" in text:
            text = os.path.basename(text)
        normalized.append(text)
    return normalized


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


def build_ragas_dataset(samples):
    return EvaluationDataset(samples=samples)


def run_responses(config: RAGConfig, api_url: str, dataset_path: Path, output_path: Path):
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)
    results = []
    ragas_samples = []

    print(f"Running responses for {len(df)} questions...")

    for i, row in df.iterrows():
        question = row.get("question") or row.get("user_input")
        ground_truth = row.get("ground_truth") or row.get("reference")
        if not question:
            continue

        db_chunk_size = getattr(config, "max_chunk_size", None) or config.chunk_size
        payload = {
            "message": question,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "db_path": f"./vector_db/db_{config.embedding_model.split('/')[-1]}_cs{db_chunk_size}",
            "use_hybrid": config.use_hybrid,
            "hybrid_weight": config.hybrid_weight,
            "use_mmr": config.use_mmr,
            "mmr_lambda": config.mmr_lambda,
            "max_output_tokens": config.max_output_tokens,
            "max_context_chars": config.max_context_chars,
            "max_doc_chars": config.max_doc_chars,
            "response_style": config.response_style,
            "use_reranker": getattr(config, "use_reranker", False),
            "rerank_top_n": getattr(config, "rerank_top_n", 8),
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
        retrieved_sources = normalize_sources(data.get("sources", []))
        retrieved_contexts = data.get("retrieved_contexts", [])
        timings = data.get("timings", {}) or {}

        reference_sources = normalize_sources(coerce_list(row.get("reference_sources", [])))
        precision, recall, matched = compute_precision_recall(retrieved_sources, reference_sources)
        reference_contexts = coerce_list(row.get("context", []))

        results.append({
            "id": f"Q_{i}",
            "question": question,
            "ground_truth": ground_truth,
            "bot_answer": bot_answer,
            "retrieved_sources": retrieved_sources,
            "retrieved_contexts": retrieved_contexts,
            "retrieved_contexts_count": len(retrieved_contexts) if retrieved_contexts else 0,
            "reference_sources": reference_sources,
            "reference_contexts": reference_contexts,
            "matched_sources": matched,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "context_precision": "",
            "context_recall": "",
            "faithfulness": "",
            "answer_relevancy": "",
            "latency_ms": round(latency_ms, 2),
            "retrieval_ms": timings.get("retrieval_ms", ""),
            "generation_ms": timings.get("generation_ms", ""),
            "total_ms": timings.get("total_ms", ""),
        })

        ragas_samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=[str(c).strip() for c in retrieved_contexts if c],
            reference_contexts=[str(c).strip() for c in reference_contexts if c],
            reference=ground_truth or "",
            response=bot_answer or "",
        ))

    if not results:
        raise RuntimeError(
            "No results were generated. Check that the API server is running and "
            "accessible, and that requests are succeeding."
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set; required for RAGAS context metrics.")

    ragas_llm = ChatGoogleGenerativeAI(
        model=config.eval_judge_model,
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True,
    )
    ragas_embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=api_key,
    )
    ragas_dataset = build_ragas_dataset(ragas_samples)
    ragas_result = evaluate(
        ragas_dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=LangchainLLMWrapper(ragas_llm),
        embeddings=ragas_embeddings,
        show_progress=True,
    )
    ragas_df = ragas_result.to_pandas()
    for idx, row in ragas_df.iterrows():
        if idx >= len(results):
            break
        ctx_precision = row.get("context_precision")
        ctx_recall = row.get("context_recall")
        faithfulness_val = row.get("faithfulness")
        answer_relevancy_val = row.get("answer_relevancy")
        if pd.isna(ctx_precision):
            ctx_precision = 0.0
        if pd.isna(ctx_recall):
            ctx_recall = 0.0
        if pd.isna(faithfulness_val):
            faithfulness_val = 0.0
        if pd.isna(answer_relevancy_val):
            answer_relevancy_val = 0.0
        results[idx]["context_precision"] = round(float(ctx_precision), 4)
        results[idx]["context_recall"] = round(float(ctx_recall), 4)
        results[idx]["faithfulness"] = round(float(faithfulness_val), 4)
        results[idx]["answer_relevancy"] = round(float(answer_relevancy_val), 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"Config saved to: {config_path}")

    # Run-level latency percentiles for quick comparison
    if results:
        df_out = pd.DataFrame(results)
        latency = pd.to_numeric(df_out.get("latency_ms"), errors="coerce").dropna()
        if not latency.empty:
            p50 = float(latency.quantile(0.50))
            p95 = float(latency.quantile(0.95))
            summary = {"latency_p50_ms": round(p50, 2), "latency_p95_ms": round(p95, 2)}
            summary_path = output_path.with_suffix(".summary.json")
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"Latency p50/p95 (ms): {summary['latency_p50_ms']} / {summary['latency_p95_ms']}")
            print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--llm-model", default=RAGConfig.llm_model)
    parser.add_argument("--embedding-model", default=RAGConfig.embedding_model)
    parser.add_argument("--temperature", type=float, default=RAGConfig.temperature)
    parser.add_argument("--chunk-size", type=int, default=RAGConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=RAGConfig.chunk_overlap)
    parser.add_argument("--min-chunk-size", type=int, default=RAGConfig.min_chunk_size)
    parser.add_argument("--max-chunk-size", type=int, default=RAGConfig.max_chunk_size)
    parser.add_argument("--top-k", type=int, default=RAGConfig.top_k)
    parser.add_argument("--use-hybrid", action="store_true", default=RAGConfig.use_hybrid)
    parser.add_argument("--no-hybrid", action="store_false", dest="use_hybrid")
    parser.add_argument("--hybrid-weight", type=float, default=RAGConfig.hybrid_weight)
    parser.add_argument("--use-mmr", action="store_true", default=RAGConfig.use_mmr)
    parser.add_argument("--no-mmr", action="store_false", dest="use_mmr")
    parser.add_argument("--mmr-lambda", type=float, default=RAGConfig.mmr_lambda)
    parser.add_argument("--use-reranker", action="store_true", default=getattr(RAGConfig, "use_reranker", False))
    parser.add_argument("--no-reranker", action="store_false", dest="use_reranker")
    parser.add_argument("--rerank-top-n", type=int, default=getattr(RAGConfig, "rerank_top_n", 8))
    parser.add_argument("--max-output-tokens", type=int, default=RAGConfig.max_output_tokens)
    parser.add_argument("--max-context-chars", type=int, default=RAGConfig.max_context_chars)
    parser.add_argument("--max-doc-chars", type=int, default=RAGConfig.max_doc_chars)
    parser.add_argument("--response-style", default=RAGConfig.response_style)
    args = parser.parse_args()

    cfg = RAGConfig(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        hybrid_weight=args.hybrid_weight,
        use_mmr=args.use_mmr,
        mmr_lambda=args.mmr_lambda,
        use_reranker=getattr(args, "use_reranker", False),
        rerank_top_n=getattr(args, "rerank_top_n", 8),
        max_output_tokens=args.max_output_tokens,
        max_context_chars=args.max_context_chars,
        max_doc_chars=args.max_doc_chars,
        response_style=args.response_style,
    )

    if args.dataset_type in {"single", "both"}:
        dataset_path = REPO_ROOT / "evals" / "singlehop_dataset.csv"
        output_path = REPO_ROOT / "results" / f"{dataset_path.stem}_responses.csv"
        run_responses(cfg, args.api_url, dataset_path, output_path)

    if args.dataset_type in {"multi", "both"}:
        dataset_path = REPO_ROOT / "evals" / "multihop_dataset.csv"
        output_path = REPO_ROOT / "results" / f"{dataset_path.stem}_responses.csv"
        run_responses(cfg, args.api_url, dataset_path, output_path)
