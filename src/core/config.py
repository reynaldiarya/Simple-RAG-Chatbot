from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"

    # Ollama configuration
    ollama_api_base: str
    ollama_api_key: SecretStr = SecretStr("")
    ollama_model: str
    ollama_temperature: float = 0.1
    ollama_top_p: float = 0.9

    # RAG configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.4
    max_retrieved_docs: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_chars: int = 4000
    max_history_messages: int = 6

    # Feature flags
    citation_enabled: bool = True

    # Storage paths
    documents_dir: str = "src/data/documents"
    vector_db_dir: str = "src/data/vector_db"

    @field_validator("ollama_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validates that the temperature is within the acceptable range."""
        if not 0.0 <= v <= 0.5:
            raise ValueError(
                "ollama_temperature must be between 0.0 and 0.5 "
                "to minimize hallucination in RAG applications."
            )
        return v

    class Config:
        env_file = ".env"


settings = Settings()
