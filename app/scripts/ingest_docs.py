import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so absolute imports work when running as a script
ROOT_DIR = Path(__file__).resolve().parents[2]
print(f"Root dir: {ROOT_DIR}")
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.services.vector_store import VectorStoreManager

def check_markdown_files(data_path="./resources/data"):
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

def load_hr_csv(csv_path="./resources/data/hr/hr_data.csv"):
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

if __name__ == "__main__":

    markdown_files = check_markdown_files()
    markdown_chunks = split_markdown_documents(markdown_files)
    print(type(markdown_chunks))
    print(markdown_chunks[12])

    # If a header-based chunk is still too big, break it down further
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    final_markdown_chunks = text_splitter.split_documents(markdown_chunks)
    print(f"Total recursive chunks created: {len(final_markdown_chunks)}")

    # HR CSV pipeline
    hr_csv_chunks = load_hr_csv()
    print(f"Total HR CSV chunks: {len(hr_csv_chunks)}")
    print(hr_csv_chunks[0])

    # Combine everything
    all_chunks = final_markdown_chunks + hr_csv_chunks

    print(f"Total final chunks: {len(all_chunks)}")

    vm = VectorStoreManager()
    vectorstore = vm.create_or_get_vectorstore(all_chunks)
