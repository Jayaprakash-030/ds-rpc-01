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
    tracker = RAGExperimentTracker(experiment_name="Checking latency")
    
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
                "--embedding-model", cfg.embedding_model,
                *(["--use-hybrid"] if getattr(cfg, "use_hybrid", False) else []),
                "--hybrid-weight", str(getattr(cfg, "hybrid_weight", 0.5)),
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
    # Define experiment sweep (run on both datasets) for new chunk-merge setup
    configs_to_test = [
        # Baselines
        {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 3, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0},
        # # Disable hybrid
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 3, "use_hybrid": False, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 5, "use_hybrid": False, "temperature": 0.0},
        # # Hybrid weight sweep
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.1, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.5, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.7, "temperature": 0.0},
        
        # # Min/max chunk size variants
        # {"chunk_size": 2500, "min_chunk_size": 800, "max_chunk_size": 1200, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 1200, "max_chunk_size": 1600, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0},
        # # Low min/max with higher top_k
        # {"chunk_size": 2500, "min_chunk_size": 400, "max_chunk_size": 600, "top_k": 10, "use_hybrid": False, "temperature": 0.0},
        # # {"chunk_size": 2500, "min_chunk_size": 400, "max_chunk_size": 600, "top_k": 15, "use_hybrid": False, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 300, "max_chunk_size": 500, "top_k": 7, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0},
        # {"chunk_size": 2500, "min_chunk_size": 300, "max_chunk_size": 500, "top_k": 8, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0},
        # # LLM model variants
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 4, "use_hybrid": True, "hybrid_weight": 0.3, "temperature": 0.0, "llm_model": "gemini-3-pro-preview"},
        # {"chunk_size": 2500, "min_chunk_size": 1000, "max_chunk_size": 1300, "top_k": 5, "use_hybrid": False, "temperature": 0.0, "llm_model": "gemini-3-pro-preview"},
    ]

    for i, params in enumerate(configs_to_test):
        cfg = RAGConfig(
            chunk_size=params["chunk_size"],
            top_k=params["top_k"],
            temperature=params["temperature"],
            llm_model=params.get("llm_model", RAGConfig.llm_model),
            use_hybrid=params.get("use_hybrid", RAGConfig.use_hybrid),
            hybrid_weight=params.get("hybrid_weight", RAGConfig.hybrid_weight),
            min_chunk_size=params.get("min_chunk_size", RAGConfig.min_chunk_size),
            max_chunk_size=params.get("max_chunk_size", RAGConfig.max_chunk_size),
        )
        temp_tag = f"T{str(params['temperature']).replace('.', '')}"
        model_tag = params.get("llm_model", RAGConfig.llm_model).replace("-", "").replace(".", "").replace("_", "")
        hybrid_tag = "HY1" if params.get("use_hybrid", RAGConfig.use_hybrid) else "HY0"
        hw_tag = f"HW{str(params.get('hybrid_weight', RAGConfig.hybrid_weight)).replace('.', '')}"
        mm_tag = f"MM{params.get('min_chunk_size', RAGConfig.min_chunk_size)}_{params.get('max_chunk_size', RAGConfig.max_chunk_size)}"
        run_name = f"Exp_{i}_CS{params['chunk_size']}_K{params['top_k']}_{temp_tag}_{model_tag}_{hybrid_tag}_{hw_tag}_{mm_tag}"
        run_experiment_pipeline(cfg, run_name, "both")
