import os
import time
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from app.config.rag_config import RAGConfig
from app.services.vector_store import VectorStoreManager


class RAGService:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vm = VectorStoreManager()
        self.vectorstore = self.vm.create_or_get_vectorstore()
        self.llm = ChatOpenAI(
            model=self.config.llm_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=self.config.temperature,
            model_kwargs={
                "max_completion_tokens": self.config.max_output_tokens,
            },
        )

    def get_response(
        self,
        query,
        user_role,
    ):
        k = self.config.top_k
        if user_role.lower() == "c-level":
            search_kwargs = {"k": k}
        else:
            search_kwargs = {
                "k": k,
                "filter": {"role": {"$in": [user_role.lower(), "general"]}}
            }

        vector_retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        allowed_roles = None
        if user_role.lower() != "c-level":
            allowed_roles = [user_role.lower(), "general"]
        bm25_retriever = self._build_bm25_retriever(
            self.vectorstore,
            k,
            allowed_roles,
        )
        if bm25_retriever is None:
            retriever = vector_retriever
        else:
            retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[
                    1 - self.config.hybrid_weight,
                    self.config.hybrid_weight,
                ],
            )

        style = (self.config.response_style or "default").lower()
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
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        t0 = time.time()
        docs = retriever.invoke(query)
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

    def _build_bm25_retriever(self, vectorstore, k: int, allowed_roles: Optional[List[str]] = None):
        try:
            store = vectorstore.get(include=["documents", "metadatas"])
        except Exception:
            return None

        documents = []
        for content, metadata in zip(store.get("documents", []), store.get("metadatas", [])):
            # Apply role-based filtering for BM25 as well, so that
            # non c-level users only see their department + general docs.
            if allowed_roles is not None:
                role = (metadata or {}).get("role")
                if role not in allowed_roles:
                    continue
            documents.append(Document(page_content=content, metadata=metadata))

        if not documents:
            return None

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        return bm25_retriever
