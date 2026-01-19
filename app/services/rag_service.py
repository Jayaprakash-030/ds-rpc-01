import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.config.experiment_config import RAGConfig
from app.services.vector_store import VectorStoreManager

class RAGService:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.vm = VectorStoreManager()
        self.vectorstore = self.vm.create_or_get_vectorstore()
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.llm_model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=self.config.temperature,  # Low temperature for factual FinTech responses
            convert_system_message_to_human=True,
        )

    def get_response(
        self,
        query,
        user_role,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        persist_directory: Optional[str] = None,
    ):
        vectorstore = self.vectorstore
        if persist_directory:
            vm = VectorStoreManager(config=self.config, persist_directory=persist_directory)
            vectorstore = vm.create_or_get_vectorstore()

        llm = self.llm
        if temperature is not None:
            llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=temperature,
                convert_system_message_to_human=True,
            )
        # 1. Define RBAC Filter
        # C-Level gets all; others get their dept + general info
        k = top_k if top_k is not None else self.config.top_k
        if user_role.lower() == "c-level":
            search_kwargs = {"k": k}
        else:
            search_kwargs = {
                "k": k,
                "filter": {"role": {"$in": [user_role.lower(), "general"]}}
            }

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        # 2. Define the Prompt (Ensures context-rich response) 
        system_prompt = (
            "You are an assistant for FinSolve Technologies. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 3. Create and Invoke the Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain.invoke({"input": query})
