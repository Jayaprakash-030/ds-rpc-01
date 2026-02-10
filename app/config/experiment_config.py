from dataclasses import dataclass, asdict

@dataclass
class RAGConfig:
    # Model Settings
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    temperature: float = 0.0
    max_output_tokens: int = 512
    response_style: str = "default"
    
    # Retrieval Settings (Exp_2: tighter chunks, hybrid, no MMR — prod default)
    chunk_size: int = 650
    chunk_overlap: int = 100
    top_k: int = 16
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    use_hybrid: bool = True
    hybrid_weight: float = 0.6
    min_chunk_size: int = 400
    max_chunk_size: int = 650
    max_context_chars: int = 0
    max_doc_chars: int = 0
    
    # Reranker (optional): improves precision by selecting top chunks for context
    use_reranker: bool = False
    rerank_top_n: int = 8
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Evaluation Settings
    eval_judge_model: str = "gemini-2.5-flash"
    
    def to_dict(self):
        """Converts config to a dictionary for MLflow logging."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Creates a config instance from a dictionary."""
        return cls(**data)
