import os
from typing import Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.services.rag_service import RAGService


app = FastAPI()
security = HTTPBasic(auto_error=False)
rag_service = RAGService()

AUTH_BYPASS = os.getenv("AUTH_BYPASS", "").lower() in {"1", "true", "yes"}

# Dummy user database
users_db: Dict[str, Dict[str, str]] = {
    "Tony": {"password": "password123", "role": "engineering"},
    "Bruce": {"password": "securepass", "role": "marketing"},
    "Sam": {"password": "financepass", "role": "finance"},
    "Peter": {"password": "pete123", "role": "engineering"},
    "Sid": {"password": "sidpass123", "role": "marketing"},
    "Natasha": {"password": "hrpass123", "role": "hr"}
}


# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials | None = Depends(security)):
    if AUTH_BYPASS:
        return {"username": "eval", "role": "c-level"}
    if credentials is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    username = credentials.username
    password = credentials.password
    user = users_db.get(username)
    if not user or user.get("password") != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": username, "role": user["role"]}


# Login endpoint
@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

# Protected chat endpoint
@app.post("/chat")
def query(user=Depends(authenticate), message: str = "Hello"):
    # user["role"] comes from the authenticate dependency in your starter code
    result = rag_service.get_response(message, user["role"])
    
    # We return the answer and the sources for citation [cite: 33]
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(doc.metadata.get("source") for doc in result["context"])),
        "role": user["role"]
    }


# Test-only endpoint (no auth)
@app.post("/chat_test")
def query_test(message: str = "Hello"):
    result = rag_service.get_response(message, "c-level")
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(doc.metadata.get("source") for doc in result["context"])),
        "role": "c-level"
    }
