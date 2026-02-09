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
EVAL_SCRIPT = REPO_ROOT / "app" / "utils" / "bot_answer_evaluation_agent.py"

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
    tracker = RAGExperimentTracker(experiment_name="Final Evaluation")
    
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
                "--min_chunk_size", str(getattr(cfg, "min_chunk_size", 1000)),
                "--max_chunk_size", str(getattr(cfg, "max_chunk_size", 1300)),
            ], "INGESTION")

            ds_arg = "both" if dataset_type == "both" else ("single" if dataset_type == "simple" else "multi")
            run_step([
                sys.executable, str(RESPONSES_SCRIPT),
                "--dataset-type", ds_arg,
                "--top-k", str(cfg.top_k),
                "--temperature", str(cfg.temperature),
                "--chunk-size", str(cfg.chunk_size),
                "--chunk-overlap", str(cfg.chunk_overlap),
                "--min-chunk-size", str(getattr(cfg, "min_chunk_size", 1000)),
                "--max-chunk-size", str(getattr(cfg, "max_chunk_size", 1300)),
                "--embedding-model", cfg.embedding_model,
                *(["--use-hybrid"] if getattr(cfg, "use_hybrid", False) else []),
                "--hybrid-weight", str(getattr(cfg, "hybrid_weight", 0.5)),
                "--max-output-tokens", str(getattr(cfg, "max_output_tokens", 512)),
                "--max-context-chars", str(getattr(cfg, "max_context_chars", 0)),
                "--max-doc-chars", str(getattr(cfg, "max_doc_chars", 0)),
                "--response-style", str(getattr(cfg, "response_style", "default")),
            ], "RESPONSES")

            datasets = ["simple", "multihop"] if dataset_type == "both" else [dataset_type]
            for ds in datasets:
                tracker.set_run_tags({"dataset": ds})
                results_file = SINGLE_RESULTS if ds == "simple" else MULTI_RESULTS
                run_step([
                    sys.executable, str(EVAL_SCRIPT),
                    "--files", str(results_file),
                    "--eval-judge-model", cfg.eval_judge_model
                ], f"EVALUATION ({ds})")

                evaluated_file = str(results_file).replace(".csv", "_automated_eval.csv")
                tracker.log_metrics_from_csv(evaluated_file, metric_prefix=ds)
            
            tracker.set_run_tags({"status": "completed"})
            print(f" Run '{run_name}' successfully completed and logged.")

        except Exception as e:
            tracker.set_run_tags({"status": "failed", "error": str(e)})
            print(f" Run '{run_name}' failed. Moving to next experiment...")

if __name__ == "__main__":
    # Define experiment sweep (run on both datasets) for MMR + Hybrid tuning
    configs_to_test = [
        # Compare two finalists
        {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.3, "use_mmr": False, "temperature": 0.0},
        {"chunk_size": 2500, "min_chunk_size": 500, "max_chunk_size": 800, "top_k": 12, "use_hybrid": True, "hybrid_weight": 0.7, "use_mmr": False, "temperature": 0.0},
    ]

    for i, params in enumerate(configs_to_test):
        cfg = RAGConfig(
            chunk_size=params["chunk_size"],
            top_k=params["top_k"],
            temperature=params["temperature"],
            llm_model=params.get("llm_model", RAGConfig.llm_model),
            use_hybrid=params.get("use_hybrid", RAGConfig.use_hybrid),
            hybrid_weight=params.get("hybrid_weight", RAGConfig.hybrid_weight),
            use_mmr=params.get("use_mmr", RAGConfig.use_mmr),
            mmr_lambda=params.get("mmr_lambda", RAGConfig.mmr_lambda),
            min_chunk_size=params.get("min_chunk_size", RAGConfig.min_chunk_size),
            max_chunk_size=params.get("max_chunk_size", RAGConfig.max_chunk_size),
            max_output_tokens=params.get("max_output_tokens", RAGConfig.max_output_tokens),
            max_context_chars=params.get("max_context_chars", RAGConfig.max_context_chars),
            max_doc_chars=params.get("max_doc_chars", RAGConfig.max_doc_chars),
            response_style=params.get("response_style", RAGConfig.response_style),
        )
        temp_tag = f"T{str(params['temperature']).replace('.', '')}"
        model_tag = params.get("llm_model", RAGConfig.llm_model).replace("-", "").replace(".", "").replace("_", "")
        hybrid_tag = "HY1" if params.get("use_hybrid", RAGConfig.use_hybrid) else "HY0"
        hw_tag = f"HW{str(params.get('hybrid_weight', RAGConfig.hybrid_weight)).replace('.', '')}"
        mm_tag = f"MM{params.get('min_chunk_size', RAGConfig.min_chunk_size)}_{params.get('max_chunk_size', RAGConfig.max_chunk_size)}"
        mmr_tag = "MR1" if params.get("use_mmr", RAGConfig.use_mmr) else "MR0"
        mrl_tag = f"MRL{str(params.get('mmr_lambda', RAGConfig.mmr_lambda)).replace('.', '')}"
        run_name = f"Exp_{i}_CS{params['chunk_size']}_K{params['top_k']}_{temp_tag}_{model_tag}_{hybrid_tag}_{hw_tag}_{mm_tag}_{mmr_tag}_{mrl_tag}"
        run_experiment_pipeline(cfg, run_name, "both")
