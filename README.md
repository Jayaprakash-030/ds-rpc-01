# DS RPC 01 — FinSolve Internal RAG Chatbot (RBAC)

Internal knowledge assistant for FinSolve Technologies with role-based access control, retrieval-augmented generation, and evaluation tooling. Built with FastAPI + Streamlit, Chroma vector storage, and LangChain + Gemini models.

## Highlights
- Role-based access control (RBAC) across engineering, finance, HR, marketing, and general content
- RAG pipeline with configurable chunking, hybrid retrieval (vector + BM25), and citations
- Local evaluation workflows (single-hop, multi-hop, golden set) with optional MLflow tracking
- Dockerized backend + frontend and local dev scripts

## Architecture
- **Backend**: FastAPI service serving `/chat` and `/chat_test`
- **Frontend**: Streamlit chat UI with Basic Auth
- **Retrieval**: Chroma vector store + optional BM25 (hybrid)
- **LLM/Embeddings**: Google Gemini via LangChain
- **Data**: Role-scoped markdown + CSV sources under `resources/data`

## Repository layout
- `app/` — FastAPI app, RAG service, and configuration
- `frontend/` — Streamlit UI
- `resources/data/` — Source documents, organized by role
- `vector_db/` — Local Chroma persistence (created by ingestion)
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
GOOGLE_API_KEY=your_api_key_here
```

Optional:
- `AUTH_BYPASS=true` to skip Basic Auth (useful for eval scripts)
- `BACKEND_URL=http://localhost:8000` for the Streamlit UI

### 3) Ingest docs (build the vector DB)
```bash
python app/scripts/ingest_docs.py \
  --chunk_size 1000 \
  --chunk_overlap 150 \
  --embedding_model models/text-embedding-004
```
This creates a Chroma DB under `vector_db/`.

### 4) Start the backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Start the frontend
```bash
streamlit run frontend/app.py
```

Open the UI at `http://localhost:8501`.

## Default users (Basic Auth)
These demo credentials are defined in `app/main.py`:
- `Tony / password123` (engineering)
- `Peter / pete123` (engineering)
- `Bruce / securepass` (marketing)
- `Sid / sidpass123` (marketing)
- `Sam / financepass` (finance)
- `Natasha / hrpass123` (hr)

Use `AUTH_BYPASS=true` for evaluation or automated runs.

## API endpoints
- `GET /login` — validate credentials
- `GET /test` — authenticated ping
- `POST /chat` — role-aware RAG response
- `POST /chat_test` — no-auth endpoint for evaluation (supports overrides)

## Configuration
RAG defaults live in `app/config/experiment_config.py`:
- `llm_model`, `embedding_model`, `temperature`
- `chunk_size`, `chunk_overlap`, `top_k`
- `use_hybrid`, `hybrid_weight`

Most scripts accept these as CLI flags to run experiments.

## Evaluation workflows
Run evaluation against datasets in `evals/`:
```bash
python app/scripts/run_responses.py --dataset-type both
python app/scripts/baseline_evaluation.py
```
Outputs land in `results/` and include a `.config.json` snapshot per run. MLflow utilities live in `app/utils/mlflow_tracker.py`.

## Docker
Build and run both services:
```bash
docker compose up --build
```
Backend: `http://localhost:8000`  
Frontend: `http://localhost:8501`

## Notes
- The vector DB path is derived from `embedding_model` and `chunk_size` for repeatable experiments.
- The RAG prompt and RBAC filtering are implemented in `app/services/rag_service.py`.

