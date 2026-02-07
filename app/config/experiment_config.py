from dataclasses import dataclass, asdict

@dataclass
class RAGConfig:
    # Model Settings
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/text-embedding-004"
    temperature: float = 0.0
    
    # Retrieval Settings
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 7
    use_hybrid: bool = False
    hybrid_weight: float = 0.5
    min_chunk_size: int = 1000
    max_chunk_size: int = 1300
    
    # Evaluation Settings
    eval_judge_model: str = "gemini-2.5-flash"
    
    def to_dict(self):
        """Converts config to a dictionary for MLflow logging."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Creates a config instance from a dictionary."""
        return cls(**data)
