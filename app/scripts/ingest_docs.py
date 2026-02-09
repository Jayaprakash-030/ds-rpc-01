import os
import sys
import argparse
from pathlib import Path

# Ensure project root is on sys.path so absolute imports work when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
print(f"Root dir: {ROOT_DIR}")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config.experiment_config import RAGConfig
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.services.vector_store import VectorStoreManager

def check_markdown_files(data_path):
    """Load markdown files while preserving header markers for splitting."""
    loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        loader_cls=TextLoader,
    )

    docs = loader.load()
    print(f"Found {len(docs)} markdown files.")

    # Uncomment for debugging
    # for doc in docs:
    #     print(f"File: {doc.metadata['source']}")
    #     print(f\"Preview: {doc.page_content[:200]}\\n\")

    return docs


def split_markdown_documents(docs):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # Keep the header text in the content for better AI context
    )

    final_chunks = []

    for doc in docs:
        content = doc.page_content
        # Extract role from the folder name (e.g., resources/data/finance/file.md -> finance)
        role = doc.metadata["source"].split("/")[-2]

        header_splits = markdown_splitter.split_text(content)

        for chunk in header_splits:
            # Metadata update: crucial for the project's security requirement
            chunk.metadata.update({
                "role": role,
                "source": os.path.basename(doc.metadata["source"]),
            })
            final_chunks.append(chunk)

    print(f"Total chunks created: {len(final_chunks)}")
    return final_chunks

def _header_key(chunk, level: int):
    key = f"Header {level}"
    return chunk.metadata.get(key)


def _merge_sequential(chunks, min_chunk_size: int, max_chunk_size: int):
    """
    Merge small header chunks within the same section, but never exceed max_chunk_size
    so that we get section-aware chunks (e.g. Q2 vs Q3 benchmarks stay separate).
    """
    merged = []
    buffer_doc = None

    for chunk in chunks:
        content = chunk.page_content or ""
        if buffer_doc is None:
            buffer_doc = chunk
            continue

        combined_len = len(buffer_doc.page_content) + 2 + len(content)
        # Merge if buffer is still small, but never exceed max_chunk_size
        if len(buffer_doc.page_content) < min_chunk_size and combined_len <= max_chunk_size:
            buffer_doc.page_content += "\n\n" + content
        else:
            merged.append(buffer_doc)
            buffer_doc = chunk

    if buffer_doc is not None:
        merged.append(buffer_doc)

    return merged


def merge_chunks_hierarchical(chunks, min_chunk_size: int, max_chunk_size: int):
    """
    Bottom-up merge:
    Merge Header 4 -> Header 3 -> Header 2 -> Header 1, never crossing parent boundaries.
    Caps merged size at max_chunk_size so sections (e.g. Q2 vs Q3) stay in separate chunks.
    """
    def group_key(chunk, level: int):
        return tuple(_header_key(chunk, i) for i in range(1, level + 1))

    merged = chunks
    for level in (4, 3, 2, 1):
        grouped = []
        buffer = []
        last_key = None

        for chunk in merged:
            key = group_key(chunk, level)
            if buffer and key != last_key:
                grouped.extend(_merge_sequential(buffer, min_chunk_size, max_chunk_size))
                buffer = []
            buffer.append(chunk)
            last_key = key

        if buffer:
            grouped.extend(_merge_sequential(buffer, min_chunk_size, max_chunk_size))

        merged = grouped

    return merged

def load_hr_csv(csv_path):
    """Load the single HR CSV file."""
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "role": "hr",
            "source": os.path.basename(csv_path),
        })

    print(f"Loaded {len(docs)} HR CSV rows.")
    return docs

def run_ingestion(config: RAGConfig, min_chunk_size: int = 1000, max_chunk_size: int = 1300):
    print(f"--- Ingesting with Min/Max Chunk Size: {min_chunk_size}/{max_chunk_size} ---")
    data_dir = ROOT_DIR / "resources" / "data"
    hr_csv_path = data_dir / "hr" / "hr_data.csv"

    markdown_files = check_markdown_files(str(data_dir))
    markdown_chunks = split_markdown_documents(markdown_files)

    # Merge small header chunks (capped at max_chunk_size for section-aware chunks)
    merged_chunks = merge_chunks_hierarchical(
        markdown_chunks, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    final_markdown_chunks = text_splitter.split_documents(merged_chunks)
    print(f"Total recursive chunks created: {len(final_markdown_chunks)}")

    # HR CSV pipeline
    hr_csv_chunks = load_hr_csv(str(hr_csv_path))
    print(f"Total HR CSV chunks: {len(hr_csv_chunks)}")

    # Combine everything
    all_chunks = final_markdown_chunks + hr_csv_chunks
    print(f"Total final chunks: {len(all_chunks)}")

    # Save to a versioned directory so we don't overwrite baselines
    model_short_name = config.embedding_model.split("/")[-1]
    db_name = f"db_{model_short_name}_cs{max_chunk_size}"
    db_path = ROOT_DIR / "vector_db" / db_name
    db_path.mkdir(parents=True, exist_ok=True)

    vm = VectorStoreManager(config=config, persist_directory=str(db_path))
    vm.create_or_get_vectorstore(all_chunks)
    return str(db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=RAGConfig.chunk_size)
    parser.add_argument("--chunk_overlap", type=int, default=RAGConfig.chunk_overlap)
    parser.add_argument("--embedding_model", type=str, default=RAGConfig.embedding_model)
    parser.add_argument("--min_chunk_size", type=int, default=RAGConfig.min_chunk_size)
    parser.add_argument("--max_chunk_size", type=int, default=RAGConfig.max_chunk_size)
    args = parser.parse_args()

    config = RAGConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, embedding_model=args.embedding_model)
    run_ingestion(
        config,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
    )
