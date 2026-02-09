import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.orchestrator import run_experiment_pipeline
from app.config.experiment_config import RAGConfig

if __name__ == "__main__":
    # Best config so far
    test_cfg = RAGConfig(
        llm_model="gemini-2.5-flash",
        embedding_model="models/gemini-embedding-001",
        chunk_size=2500,
        chunk_overlap=150,
        top_k=12,
        temperature=0.0,
        use_hybrid=True,
        hybrid_weight=0.7,
        use_mmr=False,
        mmr_lambda=0.5,
        min_chunk_size=500,
        max_chunk_size=800,
        max_context_chars=0,
        max_doc_chars=0,
    )

    print(" STARTING SMOKE TEST...")
    # This will trigger: Ingest -> Inference -> Eval -> MLflow
    run_experiment_pipeline(test_cfg, run_name="best_config_baseline", dataset_type="both")
    print(" SMOKE TEST FINISHED.")
