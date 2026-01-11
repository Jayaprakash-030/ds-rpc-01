import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load the .env relative to the repo root so it works no matter the CWD
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

class VectorStoreManager:
    def __init__(self):
        # Initialize Gemini's embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Point to a local directory to save the database 
        self.persist_directory = "./chroma_db"
        
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
