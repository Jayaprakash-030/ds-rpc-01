import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.config.rag_config import RAGConfig

# Load the .env relative to the repo root so it works no matter the CWD
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)


def _document_ids(chunks):
    occurrences = {}
    ids = []
    for chunk in chunks:
        metadata = json.dumps(chunk.metadata, sort_keys=True, default=str)
        identity = f"{metadata}\0{chunk.page_content}"
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        digest = hashlib.sha256(f"{identity}\0{occurrence}".encode()).hexdigest()
        ids.append(digest)
    return ids


class VectorStoreManager:
    def __init__(self, config: Optional[RAGConfig] = None, persist_directory: Optional[str] = None):
        self.config = config or RAGConfig()
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        # Point to a local directory to save/load the database
        if persist_directory:
            self.persist_directory = persist_directory
        elif os.getenv("PERSIST_DIRECTORY"):
            self.persist_directory = os.getenv("PERSIST_DIRECTORY")
        else:
            self.persist_directory = str(
                Path(__file__).resolve().parents[2] / "chroma_db"
            )
        
    def create_or_get_vectorstore(self, chunks=None):
        if chunks:
            ids = _document_ids(chunks)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                ids=ids,
                persist_directory=self.persist_directory,
            )
            stale_ids = set(vectorstore.get(include=[])["ids"]) - set(ids)
            if stale_ids:
                vectorstore.delete(ids=list(stale_ids))
            print(f"Successfully stored {len(chunks)} chunks in {self.persist_directory}")
        else:
            # Load the existing store from disk 
            vectorstore = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
            print("Loaded existing vector database")
            
        return vectorstore
