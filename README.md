# DS RPC 01 — FinSolve Internal RAG Chatbot (RBAC)

Internal knowledge assistant for FinSolve Technologies with role-based access control, retrieval-augmented generation, and evaluation tooling. Built with FastAPI + Streamlit, Chroma vector storage, and OpenAI models through LangChain.

## Highlights
- Role-based access control (RBAC) across engineering, finance, HR, marketing, and general content
- RAG pipeline with configurable chunking, hybrid retrieval (vector + BM25), and citations
- Dockerized backend + frontend and local dev scripts

## Architecture
- **Backend**: FastAPI service serving an authenticated `/chat` endpoint
- **Frontend**: Streamlit chat UI using JWT bearer authentication
- **Retrieval**: Chroma vector store + optional BM25 (hybrid)
- **LLM/Embeddings**: OpenAI via LangChain
- **Data**: Role-scoped markdown + CSV sources under `resources/data`

## Repository layout
- `app/` — FastAPI app, RAG service, and configuration
- `frontend/` — Streamlit UI
- `resources/data/` — Source documents, organized by role
- `chroma_db/` — Canonical production Chroma database
- `evals/` — Evaluation datasets
- `app/scripts/` — Ingestion and evaluation scripts
- `results/` — Evaluation outputs

## Quickstart (local)

### 1) Create a virtual env and install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment
Create a `.env` at the repo root:
```bash
OPENAI_API_KEY=your_api_key_here
JWT_SECRET_KEY=replace_with_a_long_random_secret
AUTH_USERS_JSON=[{"username":"Tony","password_hash":"<bcrypt-hash>","role":"engineering"}]
```

Optional:
- `BACKEND_URL=http://localhost:8000` for the Streamlit UI
- `JWT_ALGORITHM=HS256`
- `ACCESS_TOKEN_EXPIRE_MINUTES=30`

### 3) Ingest docs (build the vector DB)
```bash
python app/scripts/ingest_docs.py
```
This rebuilds the production database under `chroma_db/`.

### 4) Start the backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Start the frontend
```bash
streamlit run frontend/app.py
```

Open the UI at `http://localhost:8501`.

## API endpoints
- `POST /login` — exchange Basic credentials for a JWT
- `GET /test` — authenticated bearer-token check
- `POST /chat` — authenticated role-aware RAG response

## Configuration
Production RAG defaults live in `app/config/rag_config.py`:
- `llm_model`, `embedding_model`, `temperature`
- `chunk_size`, `chunk_overlap`, `top_k`
- `hybrid_weight`

## Docker
Build and run both services:
```bash
docker compose up --build
```
Backend: `http://localhost:8000`  
Frontend: `http://localhost:8501`

## Notes
- Local and container deployments both use the canonical `chroma_db/` database.
- The RAG prompt and RBAC filtering are implemented in `app/services/rag_service.py`.
