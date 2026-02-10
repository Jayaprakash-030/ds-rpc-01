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
    print(f"--- [STEP: {step_name}] Executing... ---")
    try:
        result = subprocess.run(command_list, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f" Error in {step_name}!")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise e


def run_experiment_pipeline(cfg: RAGConfig, run_name: str, dataset_type: str):
    tracker = RAGExperimentTracker(experiment_name="BM25_Optimization_Sprint")

    with tracker.start_run(run_name=run_name):
        tracker.log_params(cfg.to_dict())
        tracker.set_run_tags({"dataset": dataset_type, "status": "running"})

        try:
            run_step([
                sys.executable, str(INGEST_SCRIPT),
                "--chunk_size", str(cfg.chunk_size),
                "--chunk_overlap", str(cfg.chunk_overlap),
                "--embedding_model", cfg.embedding_model
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
                "--use-hybrid",
                "--hybrid-weight", str(cfg.hybrid_weight),
            ], "RESPONSES")

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
    base_cfg = {
        "llm_model": "gemini-3-pro-preview",
        "chunk_size": 750,
        "top_k": 15,
        "temperature": 0.0,
        "use_hybrid": True,
    }
    configs_to_test = [
        {**base_cfg, "hybrid_weight": 0.0},
        {**base_cfg, "hybrid_weight": 0.2},
        {**base_cfg, "hybrid_weight": 0.4},
        {**base_cfg, "hybrid_weight": 0.6},
        {**base_cfg, "hybrid_weight": 0.8},
        {**base_cfg, "hybrid_weight": 1.0},
    ]

    for i, params in enumerate(configs_to_test):
        cfg = RAGConfig(
            chunk_size=params["chunk_size"],
            top_k=params["top_k"],
            temperature=params["temperature"],
            llm_model=params.get("llm_model", RAGConfig.llm_model),
            use_hybrid=True,
            hybrid_weight=params.get("hybrid_weight", RAGConfig.hybrid_weight),
        )
        temp_tag = f"T{str(params['temperature']).replace('.', '')}"
        model_tag = params.get("llm_model", RAGConfig.llm_model).replace("-", "").replace(".", "").replace("_", "")
        weight_tag = f"HW{str(params['hybrid_weight']).replace('.', '')}"
        run_name = f"BM25_{i}_CS{params['chunk_size']}_K{params['top_k']}_{temp_tag}_{weight_tag}_{model_tag}"
        run_experiment_pipeline(cfg, run_name, "both")
