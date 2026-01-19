import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.utils.orchestrator import run_experiment_pipeline
from app.config.experiment_config import RAGConfig

if __name__ == "__main__":
    # Define a tiny, fast test config
    test_cfg = RAGConfig(
        llm_model="gemini-3-pro-preview",
        embedding_model="models/text-embedding-004",
        chunk_size=1000, 
        chunk_overlap=150,
        top_k=5,
        temperature=0.0
    )
    
    print(" STARTING SMOKE TEST...")
    # This will trigger: Ingest -> Inference -> Eval -> MLflow
    run_experiment_pipeline(test_cfg, run_name="SMOKE_TEST_03", dataset_type="both")
    print(" SMOKE TEST FINISHED.")
