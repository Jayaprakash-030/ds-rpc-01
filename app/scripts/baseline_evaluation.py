import json
from pathlib import Path

import pandas as pd
import requests

# --- Configuration ---
API_URL = "http://localhost:8000/chat_test"
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
GOLDEN_DATA_PATH = REPO_ROOT / "evals" / "golden_dataset.csv"
OUTPUT_DIR = REPO_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_quality_baseline():
    df = pd.read_csv(GOLDEN_DATA_PATH)
    results = []

    print("Starting Intelligence Baseline (42 Questions)...")

    for i, row in df.iterrows():
        payload = {"message": row["question"]}
        try:
            response = requests.post(API_URL, params=payload)
            response.raise_for_status()
            data = response.json()
            results.append({
                "id": f"Q_{i}",
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "bot_answer": data.get("answer"),
                "retrieved_chunks": data.get("sources"),
                "score_retrieval_quality": "",
                "score_llm_faithfulness": "",
                "score_answer_relevance": "",
                "error_type": ""
            })
        except Exception as exc:
            print(f"Error on Q_{i}: {exc}")

    output_path = OUTPUT_DIR / "quality_baseline_scores.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Quality Baseline complete! Open '{output_path}'.")


if __name__ == "__main__":
    run_quality_baseline()
