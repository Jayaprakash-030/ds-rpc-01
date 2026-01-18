import ast
import re
import pandas as pd
import requests
import json
import time
from pathlib import Path

# --- Configuration ---
API_URL = "http://localhost:8000/chat_test"
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
MULTIHOP_DATA_PATH = REPO_ROOT / "evals" / "multihop_dataset.csv"
OUTPUT_DIR = REPO_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_multihop_baseline():
    # 1. Load the dataset
    if not Path(MULTIHOP_DATA_PATH).exists():
        print(f" Error: {MULTIHOP_DATA_PATH} not found.")
        return

    df = pd.read_csv(MULTIHOP_DATA_PATH)
    results = []

    print(f" Running Baseline for {len(df)} Multi-Hop Questions...")

    for i, row in df.iterrows():
        # Mapping your specific column names
        # user_input -> question
        # reference -> ground_truth
        # reference_contexts -> ground_truth_contexts
        question = row['user_input']
        gt_answer = row['reference']
        gt_contexts_raw = row.get("context", "")
        try:
            gt_contexts = ast.literal_eval(gt_contexts_raw) if isinstance(gt_contexts_raw, str) else gt_contexts_raw
        except (ValueError, SyntaxError):
            gt_contexts = []
        
        payload = {"message": question}
        
        start_time = time.time()
        try:
            # Backend is mocked to 'admin' role in main.py
            response = requests.post(API_URL, params=payload)
            response.raise_for_status()
            data = response.json()
            latency = (time.time() - start_time) * 1000
            
            bot_answer = data.get("answer", "")
            bot_sources = data.get("sources", []) # List of filenames
            
            results.append({
                "id": f"MH_{i}",
                "question": question,
                "ground_truth": gt_answer,
                "bot_answer": bot_answer,
                "ground_truth_context_preview": str(gt_contexts)[:200] + "...", # For easy viewing
                "retrieved_sources": bot_sources,
                "latency_ms": round(latency, 2),
                # --- Metrics for your Manual Audit ---
                "score_retrieval_quality": "", # 1-5: Did it find the right files?
                "score_llm_faithfulness": "", # 1-5: Did it hallucinate?
                "score_reasoning_quality": "", # 1-5: Did it connect the dots?
                "error_type": "" # e.g., "Missing File A", "Logic Error"
            })
            print(f" Processed MH_{i}")
            
        except Exception as e:
            print(f" Error on Multi-Hop {i}: {e}")

    # 3. Save to a dedicated Multi-Hop results file
    output_path = OUTPUT_DIR / "multihop_baseline_scores.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    
    # 4. Print Summary Stats
    total = len(results)
    print("\n" + "="*30)
    print(f" MULTI-HOP BASELINE COMPLETE")
    print(f"Total Questions: {total}")
    print(f"Results saved to: {output_path}")
    print("="*30)

if __name__ == "__main__":
    run_multihop_baseline()
