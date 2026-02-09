import os
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from app.config.experiment_config import RAGConfig
from app.services.rag_service import RAGService


app = FastAPI()
security = HTTPBasic(auto_error=False)
rag_config = RAGConfig()
rag_service = RAGService(config=rag_config)

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
def query(
    user=Depends(authenticate),
    message: str = "Hello",
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    db_path: Optional[str] = None,
    use_hybrid: Optional[bool] = None,
    hybrid_weight: Optional[float] = None,
    use_reranker: Optional[bool] = None,
    rerank_top_n: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    max_context_chars: Optional[int] = None,
    max_doc_chars: Optional[int] = None,
    response_style: Optional[str] = None,
):
    # user["role"] comes from the authenticate dependency in your starter code
    result = rag_service.get_response(
        message,
        user["role"],
        top_k=top_k,
        temperature=temperature,
        persist_directory=db_path,
        use_hybrid=use_hybrid,
        hybrid_weight=hybrid_weight,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
        max_output_tokens=max_output_tokens,
        max_context_chars=max_context_chars,
        max_doc_chars=max_doc_chars,
        response_style=response_style,
    )
    
    # We return the answer and the sources for citation [cite: 33]
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(doc.metadata.get("source") for doc in result["context"])),
        "role": user["role"],
        "timings": result.get("timings", {}),
        "stats": result.get("stats", {}),
    }


# Test-only endpoint (no auth)
@app.post("/chat_test")
def query_test(
    message: str = "Hello",
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
    db_path: Optional[str] = None,
    use_hybrid: Optional[bool] = None,
    hybrid_weight: Optional[float] = None,
    use_mmr: Optional[bool] = None,
    mmr_lambda: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    max_context_chars: Optional[int] = None,
    max_doc_chars: Optional[int] = None,
    response_style: Optional[str] = None,
    use_reranker: Optional[bool] = None,
    rerank_top_n: Optional[int] = None,
):
    result = rag_service.get_response(
        message,
        "c-level",
        top_k=top_k,
        temperature=temperature,
        persist_directory=db_path,
        use_hybrid=use_hybrid,
        hybrid_weight=hybrid_weight,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        max_output_tokens=max_output_tokens,
        max_context_chars=max_context_chars,
        max_doc_chars=max_doc_chars,
        response_style=response_style,
        use_reranker=use_reranker,
        rerank_top_n=rerank_top_n,
    )
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(doc.metadata.get("source") for doc in result["context"])),
        "retrieved_contexts": [doc.page_content for doc in result["context"]],
        "role": "c-level",
        "timings": result.get("timings", {}),
        "stats": result.get("stats", {}),
    }
