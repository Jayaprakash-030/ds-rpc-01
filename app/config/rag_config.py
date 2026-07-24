"""Production RAG configuration."""

from dataclasses import dataclass, asdict

@dataclass
class RAGConfig:
    # Model Settings
    llm_model: str = "gpt-5.4-mini-2026-03-17"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_output_tokens: int = 512
    response_style: str = "default"
    
    # Hybrid retrieval settings
    chunk_size: int = 650
    chunk_overlap: int = 100
    top_k: int = 16
    hybrid_weight: float = 0.6
    min_chunk_size: int = 400
    max_chunk_size: int = 650
    max_context_chars: int = 0
    max_doc_chars: int = 0
    
    # Evaluation Settings
    eval_judge_model: str = "gpt-5.4-mini-2026-03-17"
    
    def to_dict(self):
        """Converts config to a dictionary for MLflow logging."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Creates a config instance from a dictionary."""
        return cls(**data)
