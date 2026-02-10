import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.experiment_config import RAGConfig
from app.utils.mlflow_tracker import RAGExperimentTracker

INGEST_SCRIPT = REPO_ROOT / "app" / "scripts" / "ingest_docs.py"
RESPONSES_SCRIPT = REPO_ROOT / "app" / "scripts" / "run_responses.py"

SINGLE_RESULTS = REPO_ROOT / "results" / "singlehop_dataset_responses.csv"
MULTI_RESULTS = REPO_ROOT / "results" / "multihop_dataset_responses.csv"


def run_step(command_list, step_name):
    """Executes a subprocess and ensures it completes successfully."""
    print(f"--- [STEP: {step_name}] Executing... ---")
    try:
        result = subprocess.run(
            command_list, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f" Error in {step_name}!")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise e

def run_experiment_pipeline(cfg: RAGConfig, run_name: str, dataset_type: str):
    """
    Orchestrates the full RAG lifecycle for a single experiment configuration.
    dataset_type: 'simple', 'multihop', or 'both'
    """
    tracker = RAGExperimentTracker(experiment_name="RAG_Optimization_RAGAS")
    
    with tracker.start_run(run_name=run_name):
        # 1. Setup metadata
        tracker.log_params(cfg.to_dict())
        tracker.set_run_tags({"dataset": dataset_type, "status": "running"})
        
        try:
            # 2. STEP 1: Ingestion (Versioned Vector Store)
            run_step([
                sys.executable, str(INGEST_SCRIPT),
                "--chunk_size", str(cfg.chunk_size),
                "--chunk_overlap", str(cfg.chunk_overlap),
                "--embedding_model", cfg.embedding_model,
                "--min_chunk_size", str(getattr(cfg, "min_chunk_size", 500)),
                "--max_chunk_size", str(getattr(cfg, "max_chunk_size", 800)),
            ], "INGESTION")

            ds_arg = "both" if dataset_type == "both" else ("single" if dataset_type == "simple" else "multi")
            response_cmd = [
                sys.executable, str(RESPONSES_SCRIPT),
                "--dataset-type", ds_arg,
                "--top-k", str(cfg.top_k),
                "--temperature", str(cfg.temperature),
                "--chunk-size", str(cfg.chunk_size),
                "--chunk-overlap", str(cfg.chunk_overlap),
                "--min-chunk-size", str(getattr(cfg, "min_chunk_size", 500)),
                "--max-chunk-size", str(getattr(cfg, "max_chunk_size", 800)),
                "--embedding-model", cfg.embedding_model,
                "--hybrid-weight", str(getattr(cfg, "hybrid_weight", 0.6)),
                "--mmr-lambda", str(getattr(cfg, "mmr_lambda", 0.5)),
                "--max-output-tokens", str(getattr(cfg, "max_output_tokens", 512)),
                "--max-context-chars", str(getattr(cfg, "max_context_chars", 0)),
                "--max-doc-chars", str(getattr(cfg, "max_doc_chars", 0)),
                "--response-style", str(getattr(cfg, "response_style", "default")),
            ]
            if getattr(cfg, "use_hybrid", True):
                response_cmd.append("--use-hybrid")
            else:
                response_cmd.append("--no-hybrid")
            if getattr(cfg, "use_mmr", True):
                response_cmd.append("--use-mmr")
            else:
                response_cmd.append("--no-mmr")
            if getattr(cfg, "use_reranker", False):
                response_cmd.extend(["--use-reranker", "--rerank-top-n", str(getattr(cfg, "rerank_top_n", 8))])
            run_step(response_cmd, "RESPONSES")

            # Log RAGAS metrics from response CSVs (no separate bot evaluation step)
            datasets = ["simple", "multihop"] if dataset_type == "both" else [dataset_type]
            for ds in datasets:
                tracker.set_run_tags({"dataset": ds})
                results_file = SINGLE_RESULTS if ds == "simple" else MULTI_RESULTS
                tracker.log_metrics_from_csv(str(results_file), metric_prefix=ds)

            tracker.set_run_tags({"status": "completed"})
            print(f" Run '{run_name}' successfully completed and logged.")

        except Exception as e:
            tracker.set_run_tags({"status": "failed", "error": str(e)})
            print(f" Run '{run_name}' failed. Moving to next experiment...")

if __name__ == "__main__":
    # ---------- Phase 1 (commented out; uncomment to re-run hybrid-only baseline) ----------
    # configs_to_test = [
    #     # Baseline
    #     {"min_chunk_size": 500, "max_chunk_size": 800, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": False, "use_reranker": False, "temperature": 0.0},
    #     # More candidates
    #     {"min_chunk_size": 500, "max_chunk_size": 800, "top_k": 20, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": False, "use_reranker": False, "temperature": 0.0},
    #     # Tighter chunks (Exp_2)
    #     {"min_chunk_size": 400, "max_chunk_size": 650, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": False, "use_reranker": False, "temperature": 0.0},
    #     # More semantic (higher hybrid_weight)
    #     {"min_chunk_size": 500, "max_chunk_size": 800, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.7, "use_mmr": False, "use_reranker": False, "temperature": 0.0},
    #     # High chunk + low top_k
    #     {"min_chunk_size": 700, "max_chunk_size": 1100, "top_k": 6, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": False, "use_reranker": False, "temperature": 0.0},
    #     {"min_chunk_size": 700, "max_chunk_size": 1100, "top_k": 8, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": False, "use_reranker": False, "temperature": 0.0},
    # ]

    # ---------- Phase 2: MMR sweep on Exp_2 (400-650, K16) and Exp_5 (700-1100, K8) ----------
    configs_to_test = [
        # Exp_2 (tighter chunks) + MMR — lambda sweep
        {"min_chunk_size": 400, "max_chunk_size": 650, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": True, "mmr_lambda": 0.4, "use_reranker": False, "temperature": 0.0},
        {"min_chunk_size": 400, "max_chunk_size": 650, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": True, "mmr_lambda": 0.5, "use_reranker": False, "temperature": 0.0},
        {"min_chunk_size": 400, "max_chunk_size": 650, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": True, "mmr_lambda": 0.6, "use_reranker": False, "temperature": 0.0},
        # Exp_2 + slightly more semantic with MMR
        {"min_chunk_size": 400, "max_chunk_size": 650, "top_k": 16, "use_hybrid": True, "hybrid_weight": 0.65, "use_mmr": True, "mmr_lambda": 0.5, "use_reranker": False, "temperature": 0.0},
        # Exp_5 (high chunk, K8) + MMR — lambda sweep
        {"min_chunk_size": 700, "max_chunk_size": 1100, "top_k": 8, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": True, "mmr_lambda": 0.4, "use_reranker": False, "temperature": 0.0},
        {"min_chunk_size": 700, "max_chunk_size": 1100, "top_k": 8, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": True, "mmr_lambda": 0.5, "use_reranker": False, "temperature": 0.0},
        {"min_chunk_size": 700, "max_chunk_size": 1100, "top_k": 8, "use_hybrid": True, "hybrid_weight": 0.6, "use_mmr": True, "mmr_lambda": 0.6, "use_reranker": False, "temperature": 0.0},
        # Exp_5 + more semantic, higher diversity (lambda 0.55)
        {"min_chunk_size": 700, "max_chunk_size": 1100, "top_k": 8, "use_hybrid": True, "hybrid_weight": 0.65, "use_mmr": True, "mmr_lambda": 0.55, "use_reranker": False, "temperature": 0.0},
    ]

    for i, params in enumerate(configs_to_test):
        chunk_size = params.get("chunk_size", params.get("max_chunk_size", RAGConfig.chunk_size))
        cfg = RAGConfig(
            chunk_size=chunk_size,
            top_k=params["top_k"],
            temperature=params.get("temperature", RAGConfig.temperature),
            llm_model=params.get("llm_model", RAGConfig.llm_model),
            use_hybrid=params.get("use_hybrid", RAGConfig.use_hybrid),
            hybrid_weight=params.get("hybrid_weight", RAGConfig.hybrid_weight),
            use_mmr=params.get("use_mmr", RAGConfig.use_mmr),
            mmr_lambda=params.get("mmr_lambda", RAGConfig.mmr_lambda),
            min_chunk_size=params.get("min_chunk_size", RAGConfig.min_chunk_size),
            max_chunk_size=params.get("max_chunk_size", RAGConfig.max_chunk_size),
            use_reranker=params.get("use_reranker", getattr(RAGConfig, "use_reranker", False)),
            rerank_top_n=params.get("rerank_top_n", getattr(RAGConfig, "rerank_top_n", 8)),
            max_output_tokens=params.get("max_output_tokens", RAGConfig.max_output_tokens),
            max_context_chars=params.get("max_context_chars", RAGConfig.max_context_chars),
            max_doc_chars=params.get("max_doc_chars", RAGConfig.max_doc_chars),
            response_style=params.get("response_style", RAGConfig.response_style),
        )
        temp_tag = f"T{str(params.get('temperature', 0)).replace('.', '')}"
        model_tag = params.get("llm_model", RAGConfig.llm_model).replace("-", "").replace(".", "").replace("_", "")
        hybrid_tag = "HY1" if params.get("use_hybrid", RAGConfig.use_hybrid) else "HY0"
        hw_tag = f"HW{str(params.get('hybrid_weight', RAGConfig.hybrid_weight)).replace('.', '')}"
        mm_tag = f"MM{params.get('min_chunk_size', RAGConfig.min_chunk_size)}_{params.get('max_chunk_size', RAGConfig.max_chunk_size)}"
        mmr_tag = "MR1" if params.get("use_mmr", RAGConfig.use_mmr) else "MR0"
        rr_tag = "RR1" if params.get("use_reranker", False) else "RR0"
        run_name = f"Exp_{i}_K{params['top_k']}_{temp_tag}_{hybrid_tag}_{hw_tag}_{mm_tag}_{mmr_tag}_{rr_tag}"
        run_experiment_pipeline(cfg, run_name, "both")
