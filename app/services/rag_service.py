import os
import time
import math
from typing import Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from app.config.experiment_config import RAGConfig
from app.services.vector_store import VectorStoreManager

# Optional reranker (sentence-transformers); skip if not installed
def _get_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except ImportError:
        return None

class RAGService:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vm = VectorStoreManager()
        self.vectorstore = self.vm.create_or_get_vectorstore()
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=self.config.temperature,  # Low temperature for factual FinTech responses
            max_output_tokens=self.config.max_output_tokens,
            convert_system_message_to_human=True,
        )

    def get_response(
        self,
        query,
        user_role,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        persist_directory: Optional[str] = None,
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
        vectorstore = self.vectorstore
        if persist_directory:
            vm = VectorStoreManager(config=self.config, persist_directory=persist_directory)
            vectorstore = vm.create_or_get_vectorstore()

        resolved_temperature = self.config.temperature if temperature is None else temperature
        resolved_max_output_tokens = (
            self.config.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        llm = self.llm
        if temperature is not None or max_output_tokens is not None:
            llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=resolved_temperature,
                max_output_tokens=resolved_max_output_tokens,
                convert_system_message_to_human=True,
            )
        # Optional query cleanup for noisy/typo-heavy inputs
        # 1. Define RBAC Filter
        # C-Level gets all; others get their dept + general info
        k = top_k if top_k is not None else self.config.top_k
        use_hybrid = self.config.use_hybrid if use_hybrid is None else use_hybrid
        hybrid_weight = self.config.hybrid_weight if hybrid_weight is None else hybrid_weight
        use_mmr = self.config.use_mmr if use_mmr is None else use_mmr
        mmr_lambda = self.config.mmr_lambda if mmr_lambda is None else mmr_lambda
        use_reranker = self.config.use_reranker if use_reranker is None else use_reranker
        rerank_top_n = self.config.rerank_top_n if rerank_top_n is None else rerank_top_n
        if user_role.lower() == "c-level":
            search_kwargs = {"k": k}
        else:
            search_kwargs = {
                "k": k,
                "filter": {"role": {"$in": [user_role.lower(), "general"]}}
            }

        # Fetch more candidates when using MMR so we can diversify (e.g. multiple docs for multihop)
        fetch_k = max(k * 3, 20) if use_mmr and use_hybrid else (max(k * 2, 10) if use_mmr else k)
        search_kwargs["k"] = fetch_k

        if use_mmr and not use_hybrid:
            mmr_kwargs = dict(search_kwargs)
            mmr_kwargs["fetch_k"] = max(k * 2, 10)
            mmr_kwargs["lambda_mult"] = mmr_lambda
            vector_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=mmr_kwargs)
        else:
            vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        if use_hybrid:
            bm25_retriever = self._build_bm25_retriever(vectorstore, fetch_k)
            if os.getenv("RAG_DEBUG_RETRIEVER", "").lower() in {"1", "true", "yes"}:
                try:
                    vec_docs = vector_retriever.get_relevant_documents(query)
                    bm25_docs = bm25_retriever.get_relevant_documents(query)
                    combined_ids = {id(d) for d in vec_docs} | {id(d) for d in bm25_docs}
                    print(
                        f"[retriever_debug] vector={len(vec_docs)} bm25={len(bm25_docs)} "
                        f"unique={len(combined_ids)} k={k}"
                    )
                except Exception as exc:
                    print(f"[retriever_debug] failed to collect counts: {exc}")
            retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[1 - hybrid_weight, hybrid_weight],
            )
        else:
            if os.getenv("RAG_DEBUG_RETRIEVER", "").lower() in {"1", "true", "yes"}:
                try:
                    vec_docs = vector_retriever.get_relevant_documents(query)
                    print(f"[retriever_debug] vector={len(vec_docs)} k={k}")
                except Exception as exc:
                    print(f"[retriever_debug] failed to collect counts: {exc}")
            retriever = vector_retriever

        # 2. Define the Prompt (Ensures context-rich response) 
        style = (response_style or self.config.response_style or "default").lower()
        concise_instruction = (
            "Answer concisely in 3-6 sentences unless the user requests more detail. "
        )
        system_prompt = (
            "You are an assistant for FinSolve Technologies. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            + (concise_instruction if style in {"default", "concise"} else "")
            + "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 3. Retrieve docs and invoke the chain with timing
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        t0 = time.time()
        docs = retriever.get_relevant_documents(query)
        if use_hybrid and use_mmr:
            docs = self._mmr_select(query, docs, k, mmr_lambda, vectorstore)
        if use_reranker and docs:
            docs = self._rerank(query, docs, rerank_top_n)
        t1 = time.time()

        result = question_answer_chain.invoke({"input": query, "context": docs})
        t2 = time.time()
        answer = result.get("answer") if isinstance(result, dict) else result
        answer_text = str(answer) if answer is not None else ""
        context_chars = sum(len(doc.page_content or "") for doc in docs)

        return {
            "answer": answer,
            "context": docs,
            "timings": {
                "retrieval_ms": round((t1 - t0) * 1000, 2),
                "generation_ms": round((t2 - t1) * 1000, 2),
                "total_ms": round((t2 - t0) * 1000, 2),
            },
            "stats": {
                "context_chars": context_chars,
                "context_docs": len(docs),
                "answer_chars": len(answer_text),
            },
        }

    def _mmr_select(self, query, docs, k: int, lambda_mult: float, vectorstore):
        if not docs or k <= 0:
            return []
        if len(docs) <= k:
            return docs

        embedder = getattr(vectorstore, "_embedding_function", None) or getattr(
            vectorstore, "embedding_function", None
        )
        if embedder is None:
            return docs[:k]

        query_emb = embedder.embed_query(query)
        doc_texts = [d.page_content or "" for d in docs]
        doc_embs = embedder.embed_documents(doc_texts)

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        sims = [cosine(query_emb, emb) for emb in doc_embs]
        selected = []
        selected_idx = []
        remaining = set(range(len(docs)))

        # Pick the most relevant first
        first = max(remaining, key=lambda i: sims[i])
        selected.append(docs[first])
        selected_idx.append(first)
        remaining.remove(first)

        while remaining and len(selected) < k:
            def mmr_score(i):
                max_sim = max(cosine(doc_embs[i], doc_embs[j]) for j in selected_idx)
                return lambda_mult * sims[i] - (1 - lambda_mult) * max_sim

            next_idx = max(remaining, key=mmr_score)
            selected.append(docs[next_idx])
            selected_idx.append(next_idx)
            remaining.remove(next_idx)

        return selected

    def _rerank(self, query: str, docs: List[Document], top_n: int) -> List[Document]:
        """Rerank retrieved docs with a cross-encoder and return top_n for better context precision."""
        model_name = getattr(self.config, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        encoder = _get_reranker(model_name)
        if encoder is None:
            return docs[:top_n]
        pairs = [(query, d.page_content or "") for d in docs]
        scores = encoder.predict(pairs)
        try:
            import numpy as np
            if isinstance(scores, np.ndarray):
                scores = scores.flatten().tolist()
        except ImportError:
            pass
        if not isinstance(scores, list):
            scores = list(scores)
        indexed = [(float(scores[i]), i) for i in range(len(docs))]
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [docs[i] for _, i in indexed[:top_n]]

    def _build_bm25_retriever(self, vectorstore, k: int):
        try:
            store = vectorstore.get(include=["documents", "metadatas"])
        except Exception:
            return BM25Retriever.from_documents([], k=k)

        documents = []
        for content, metadata in zip(store.get("documents", []), store.get("metadatas", [])):
            documents.append(Document(page_content=content, metadata=metadata))

        if not documents:
            return BM25Retriever.from_documents([], k=k)

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        return bm25_retriever
