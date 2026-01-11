import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.services.vector_store import VectorStoreManager

class RAGService:
    def __init__(self):
        self.vm = VectorStoreManager()
        self.vectorstore = self.vm.create_or_get_vectorstore()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,  # Low temperature for factual FinTech responses
            convert_system_message_to_human=True,
        )

    def get_response(self, query, user_role):
        # 1. Define RBAC Filter
        # C-Level gets all; others get their dept + general info
        if user_role.lower() == "c-level":
            search_kwargs = {"k": 5}
        else:
            search_kwargs = {
                "k": 5,
                "filter": {"role": {"$in": [user_role.lower(), "general"]}}
            }

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

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

        # 3. Create and Invoke the Chain [cite: 13]
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain.invoke({"input": query})
