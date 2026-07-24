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
from ragas.testset.synthesizers import MultiHopSpecificQuerySynthesizer, SingleHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Patch the event loop for RAGAS async generation.
nest_asyncio.apply()

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

# 2. Setup OpenAI
llm_instance = ChatOpenAI(
    model="gpt-5.4-mini-2026-03-17",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
)
emb_instance = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# 3. Load Docs
def load_docs():
    data_dir = REPO_ROOT / "resources" / "data"
    include_dirs = ["engineering", "finance", "hr", "marketing", "general"]
    docs = []
    for subdir in include_dirs:
        loader = DirectoryLoader(data_dir / subdir, glob="**/*.md", loader_cls=TextLoader)
        docs.extend(loader.load())
    for doc in docs:
        doc.metadata["filename"] = os.path.basename(doc.metadata.get("source", "unknown"))
    return docs

documents = load_docs()

# 4. Generator Setup
generator = TestsetGenerator.from_langchain(
    llm=llm_instance,
    embedding_model=emb_instance
)

# 5. Query distributions
multi_query_distribution = [
    (MultiHopSpecificQuerySynthesizer(llm=generator.llm), 1.0)
]
single_query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator.llm), 1.0)
]

async def generate(testset_size, query_distribution, label):
    print(f" Generating {testset_size} {label} cases...")
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=testset_size,
        query_distribution=query_distribution,
        raise_exceptions=False,
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

def build_doc_index(docs):
    index = []
    for doc in docs:
        normalized = normalize_ws(doc.page_content)
        index.append((doc.metadata.get("filename"), normalized))
    return index


def map_contexts_to_sources(contexts, doc_index):
    sources = []
    for ctx in contexts:
        ctx_text = normalize_ws(str(ctx))
        ctx_text = ctx_text.replace("<1-hop>", "").replace("<2-hop>", "").strip()
        snippet = ctx_text[:200]
        match = None
        for filename, normalized in doc_index:
            if snippet and snippet in normalized:
                match = filename
                break
        sources.append(match or "unknown")
    return sources

# Run the async generation
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # Generate datasets (oversample to handle filtering)
    multi_df = loop.run_until_complete(generate(60, multi_query_distribution, "multi-hop"))
    single_df = loop.run_until_complete(generate(80, single_query_distribution, "single-hop"))
    
    # Normalize context column name across Ragas versions
    def normalize_context_column(df):
        if "contexts" in df.columns:
            df = df.rename(columns={"contexts": "context"})
        elif "reference_contexts" in df.columns:
            df = df.rename(columns={"reference_contexts": "context"})
        return df

    multi_df = normalize_context_column(multi_df)
    single_df = normalize_context_column(single_df)

    multi_df["user_input"] = multi_df["user_input"].astype(str).map(normalize_ws)
    single_df["user_input"] = single_df["user_input"].astype(str).map(normalize_ws)

    print(f"Generated multi-hop rows: {len(multi_df)}")
    multi_df = multi_df[multi_df["user_input"].map(question_is_complex)]
    print(f"Multi-hop after question filter: {len(multi_df)}")

    multi_df["context"] = multi_df["context"].map(coerce_contexts)
    multi_df = multi_df[multi_df["context"].map(context_is_multihop)]
    print(f"Multi-hop after context filter: {len(multi_df)}")

    multi_df = multi_df[
        multi_df.apply(lambda r: answer_uses_multiple_contexts(str(r.get("reference", "")), r.get("context", [])), axis=1)
    ]
    print(f"Multi-hop after answer filter: {len(multi_df)}")

    doc_index = build_doc_index(documents)
    multi_df["reference_sources"] = multi_df["context"].map(lambda c: map_contexts_to_sources(c, doc_index))
    single_df["context"] = single_df["context"].map(coerce_contexts)
    single_df["reference_sources"] = single_df["context"].map(lambda c: map_contexts_to_sources(c, doc_index))

    # Trim to requested sizes
    multi_target = 30
    single_target = 50
    if len(multi_df) < multi_target:
        print(f"Warning: only {len(multi_df)} multi-hop rows after filtering.")
    if len(single_df) < single_target:
        print(f"Warning: only {len(single_df)} single-hop rows generated.")

    multi_df = multi_df.head(multi_target)
    single_df = single_df.head(single_target)

    multihop_path = REPO_ROOT / "evals" / "multihop_dataset.csv"
    singlehop_path = REPO_ROOT / "evals" / "singlehop_dataset.csv"
    multi_df.to_csv(multihop_path, index=False)
    single_df.to_csv(singlehop_path, index=False)

    print(f" Multi-hop questions saved to {multihop_path}")
    print(f" Single-hop questions saved to {singlehop_path}")
