import subprocess
import sys
from pathlib import Path

from app.config.experiment_config import RAGConfig
from app.utils.mlflow_tracker import RAGExperimentTracker

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

INGEST_SCRIPT = REPO_ROOT / "app" / "scripts" / "ingest_docs.py"
BASELINE_SCRIPT = REPO_ROOT / "app" / "scripts" / "baseline_evaluation.py"
MULTIHOP_SCRIPT = REPO_ROOT / "app" / "scripts" / "multihop_dataset_responses.py"
EVAL_SCRIPT = REPO_ROOT / "app" / "utils" / "bot_answer_evaluation_agent.py"

BASELINE_RESULTS = REPO_ROOT / "results" / "quality_baseline_scores.csv"
MULTIHOP_RESULTS = REPO_ROOT / "results" / "multihop_baseline_scores.csv"


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
    tracker = RAGExperimentTracker(experiment_name="RAG_Optimization_Sprint")
    
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
                "--embedding_model", cfg.embedding_model
            ], "INGESTION")

            datasets = ["simple", "multihop"] if dataset_type == "both" else [dataset_type]

            for ds in datasets:
                # 3. STEP 2: Inference (Inference on Dataset)
                inference_script = BASELINE_SCRIPT if ds == "simple" else MULTIHOP_SCRIPT
                
                tracker.set_run_tags({"dataset": ds})
                run_step([
                    sys.executable, str(inference_script),
                    "--top-k", str(cfg.top_k),
                    "--temperature", str(cfg.temperature),
                    "--chunk-size", str(cfg.chunk_size),
                    "--chunk-overlap", str(cfg.chunk_overlap),
                    "--embedding-model", cfg.embedding_model,
                    "--run-name", run_name
                ], f"INFERENCE ({ds})")

                # 4. STEP 3: Automated Evaluation (The Judge)
                base_results = BASELINE_RESULTS if ds == "simple" else MULTIHOP_RESULTS
                results_file = base_results.with_name(
                    f"{base_results.stem}_{run_name}{base_results.suffix}"
                )
                
                run_step([
                    sys.executable, str(EVAL_SCRIPT),
                    "--files", str(results_file),
                    "--eval-judge-model", cfg.eval_judge_model
                ], f"EVALUATION ({ds})")

                # 5. STEP 4: MLflow Logging
                evaluated_file = str(results_file).replace(".csv", "_automated_eval.csv")
                tracker.log_metrics_from_csv(evaluated_file, metric_prefix=ds)
            
            tracker.set_run_tags({"status": "completed"})
            print(f" Run '{run_name}' successfully completed and logged.")

        except Exception as e:
            tracker.set_run_tags({"status": "failed", "error": str(e)})
            print(f" Run '{run_name}' failed. Moving to next experiment...")

if __name__ == "__main__":
    # Define your Experiment Sweep (Hyperparameter Grid)
    # Experiment 01: Testing if larger chunks fix Multi-Hop reasoning
    configs_to_test = [
        {"chunk_size": 1000, "top_k": 5, "type": "multihop"},
        {"chunk_size": 1000, "top_k": 10, "type": "multihop"},
        {"chunk_size": 1500, "top_k": 7, "type": "multihop"}
    ]

    for i, params in enumerate(configs_to_test):
        cfg = RAGConfig(
            chunk_size=params["chunk_size"],
            top_k=params["top_k"]
        )
        run_name = f"Exp_{i}_CS{params['chunk_size']}_K{params['top_k']}"
        run_experiment_pipeline(cfg, run_name, params["type"])
