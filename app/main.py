from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPAuthorizationCredentials, HTTPBearer

from app.auth import authenticate_user, create_access_token, decode_access_token
from app.config.rag_config import RAGConfig
from app.services.rag_service import RAGService


app = FastAPI()
security = HTTPBasic(auto_error=False)
bearer_security = HTTPBearer(auto_error=False)
rag_config = RAGConfig()
rag_service = RAGService(config=rag_config)

# Authentication dependency
def authenticate(credentials: HTTPBasicCredentials | None = Depends(security)):
    if credentials is None:
        raise HTTPException(401, detail="Invalid credentials")
    try:
        user = authenticate_user(credentials.username, credentials.password)
    except ValueError:
        raise HTTPException(status_code=500, detail="Authentication is not configured")
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"username": user.username, "role": user.role}

def authenticate_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_security)):
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user = decode_access_token(credentials.credentials)
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="Authentication is not configured",
        )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user

# Login endpoint
@app.post("/login")
def login(user=Depends(authenticate)):
    access_token = create_access_token(user['username'], user['role'])
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

# Protected test endpoint
@app.get("/test")
def test(user=Depends(authenticate_token)):
    return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}

class ChatRequest(BaseModel):
    message: str

# Protected production chat endpoint
@app.post("/chat")
def query(
    message: ChatRequest,
    user=Depends(authenticate_token),
):
    result = rag_service.get_response(
        message.message,
        user_role=user["role"],
    )
    # Response for frontend: answer + sources (and role for display)
    return {
        "answer": result["answer"],
        "sources": list(dict.fromkeys(doc.metadata.get("source") for doc in result["context"])),
        "role": user["role"],
    }
