import os
import re
import ast
import pandas as pd
import nest_asyncio
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# --- Ragas & LangChain ---
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import MultiHopSpecificQuerySynthesizer
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 1. FIX: Patch the event loop for Google/Ragas async conflict
nest_asyncio.apply()

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

# 2. Setup Gemini
llm_instance = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
emb_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 3. Load Docs
def load_docs():
    data_dir = REPO_ROOT / "resources" / "data"
    include_dirs = ["engineering", "finance", "hr", "marketing", "general"]
    docs = []
    for subdir in include_dirs:
        loader = DirectoryLoader(data_dir / subdir, glob="**/*.md", loader_cls=TextLoader)
        docs.extend(loader.load())
    for doc in docs:
        doc.metadata["filename"] = doc.metadata.get("source", "unknown")
    return docs

documents = load_docs()

# 4. Generator Setup
generator = TestsetGenerator.from_langchain(
    llm=llm_instance,
    embedding_model=emb_instance
)

# 5. Focus ONLY on Multi-Hop (Complex reasoning)
query_distribution = [
    (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 1.0)
]

async def generate():
    print(" Generating 60 advanced Multi-Hop cases...")
    testset = generator.generate_with_langchain_docs(
        documents, 
        testset_size=60, 
        query_distribution=query_distribution
    )
    return testset.to_pandas()

# 6. Post-process filters to improve quality
QUESTION_MAX_CHARS = 320
MIN_KEY_TERMS = 1
MIN_CONTEXTS = 2
MAX_CONTEXT_CHARS = 12000

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def question_is_complex(q: str) -> bool:
    q_low = q.lower()
    key_terms = ["and", "also", "both", "compare", "relation", "difference", "how", "why", "impact"]
    if len(q) < 25 or len(q) > QUESTION_MAX_CHARS:
        return False
    hits = sum(1 for term in key_terms if term in q_low)
    return hits >= MIN_KEY_TERMS

def context_is_multihop(contexts: list[str]) -> bool:
    if not isinstance(contexts, list):
        return False
    if len(contexts) < MIN_CONTEXTS:
        return False
    total_chars = sum(len(c) for c in contexts)
    return total_chars <= MAX_CONTEXT_CHARS

def answer_uses_multiple_contexts(answer: str, contexts: list[str]) -> bool:
    if not answer or not contexts or len(contexts) < MIN_CONTEXTS:
        return False
    answer_low = answer.lower()
    hits = 0
    for ctx in contexts[:3]:
        ctx_tokens = [w for w in re.findall(r"[a-zA-Z0-9]{4,}", ctx)][:80]
        if any(tok.lower() in answer_low for tok in ctx_tokens):
            hits += 1
    return hits >= 1

def coerce_contexts(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            return []
    return []

# Run the async generation
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    new_multi_hop_df = loop.run_until_complete(generate())

    # 6. Append to your existing 42 QA pairs
    existing_df = pd.read_csv(REPO_ROOT / "evals" / "golden_dataset.csv")
    
    # Normalize context column name across Ragas versions
    if "contexts" in new_multi_hop_df.columns:
        new_multi_hop_df = new_multi_hop_df.rename(columns={"contexts": "context"})
    elif "reference_contexts" in new_multi_hop_df.columns:
        new_multi_hop_df = new_multi_hop_df.rename(columns={"reference_contexts": "context"})

    new_multi_hop_df["user_input"] = new_multi_hop_df["user_input"].astype(str).map(normalize_ws)
    print(f"Generated rows: {len(new_multi_hop_df)}")

    new_multi_hop_df = new_multi_hop_df[
        new_multi_hop_df["user_input"].map(question_is_complex)
    ]
    print(f"After question filter: {len(new_multi_hop_df)}")

    new_multi_hop_df["context"] = new_multi_hop_df["context"].map(coerce_contexts)
    new_multi_hop_df = new_multi_hop_df[
        new_multi_hop_df["context"].map(context_is_multihop)
    ]
    print(f"After context filter: {len(new_multi_hop_df)}")

    new_multi_hop_df = new_multi_hop_df[
        new_multi_hop_df.apply(lambda r: answer_uses_multiple_contexts(str(r.get("reference", "")), r.get("context", [])), axis=1)
    ]
    print(f"After answer filter: {len(new_multi_hop_df)}")
    
    multihop_path = REPO_ROOT / "evals" / "multihop_dataset.csv"
    new_multi_hop_df.to_csv(multihop_path, index=False)
    
    print(f" Multi-hop questions saved to {multihop_path}")
