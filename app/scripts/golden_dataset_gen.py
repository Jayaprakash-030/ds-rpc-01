import os
import time
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import google.generativeai as genai

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

# --- CONFIGURATION ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = str(ROOT_DIR / "resources" / "data")
OUTPUT_FILE = str(ROOT_DIR / "evals" / "golden_dataset.csv")

# Professional System Prompt for high-quality Q&A
PROMPT_TEMPLATE = """
You are a Senior AI Data Engineer. Your task is to generate high-quality evaluation pairs for a RAG system.
Based on the provided CONTEXT, generate a QUESTION that an employee would actually ask and an authoritative GROUND_TRUTH answer.

CONTEXT:
{context}

STRICT OUTPUT FORMAT (CSV-FRIENDLY):
QUESTION: <One-line specific question>
GROUND_TRUTH: <Detailed, accurate answer based ONLY on the context>
"""

def generate_qa_pair(context_text):
    """Generates a Q&A pair with a small delay to respect API rate limits."""
    time.sleep(1) # Rate limiting for Gemini Flash
    response = model.generate_content(PROMPT_TEMPLATE.format(context=context_text))
    
    try:
        text = response.text
        question = text.split("QUESTION:")[1].split("GROUND_TRUTH:")[0].strip()
        answer = text.split("GROUND_TRUTH:")[1].strip()
        return question, answer
    except:
        return None, None

def process_files():
    golden_dataset = []

    # Use os.walk to search ALL subdirectories
    for root, dirs, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # Skip hidden files like .DS_Store
            if filename.startswith('.'):
                continue

            print(f"--- Found and Processing: {filename} ---")

            # 1. Handle CSV Files
            if filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    for i, row in df.head(10).iterrows():
                        context = " | ".join([f"{col}: {val}" for col, val in row.items()])
                        q, a = generate_qa_pair(context)
                        if q: golden_dataset.append({"question": q, "ground_truth": a, "source": filename})
                except Exception as e:
                    print(f"Error reading CSV {filename}: {e}")

            # 2. Handle Markdown Files
            elif filename.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    sections = content.split('##') 
                    for section in sections[1:6]:
                        if len(section.strip()) > 50: # Only process substantial sections
                            q, a = generate_qa_pair(section.strip())
                            if q: golden_dataset.append({"question": q, "ground_truth": a, "source": filename})

    # Save Results
    if golden_dataset:
        output_path = Path(OUTPUT_FILE)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(golden_dataset).to_csv(output_path, index=False)
        print(f"\n SUCCESS: Found files in subfolders and generated {len(golden_dataset)} pairs.")
    else:
        print(" Still no data. Check if your DATA_PATH matches your folder name exactly.")

if __name__ == "__main__":
    process_files()

