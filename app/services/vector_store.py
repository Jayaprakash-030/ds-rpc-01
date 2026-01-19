import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config.experiment_config import RAGConfig

# Load the .env relative to the repo root so it works no matter the CWD
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

class VectorStoreManager:
    def __init__(self, config: Optional[RAGConfig] = None, persist_directory: Optional[str] = None):
        self.config = config or RAGConfig()
        # Initialize Gemini's embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Point to a local directory to save the database
        if persist_directory:
            self.persist_directory = persist_directory
        else:
            model_short_name = self.config.embedding_model.split("/")[-1]
            self.persist_directory = f"./vector_db/db_{model_short_name}_ch_{self.config.chunk_size}"
        
    def create_or_get_vectorstore(self, chunks=None):
        if chunks:
            # Create a new store from chunks and save to disk 
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"Successfully stored {len(chunks)} chunks in {self.persist_directory}")
        else:
            # Load the existing store from disk 
            vectorstore = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=self.embeddings
            )
            print("Loaded existing vector database")
            
        return vectorstore
