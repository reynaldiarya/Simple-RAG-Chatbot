from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # Ollama configuration
    ollama_api_base: str
    ollama_api_key: str
    ollama_model: str
    ollama_temperature: float = 0.1
    ollama_top_p: float = 0.9
    
    # RAG configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.4
    max_retrieved_docs: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Storage paths
    documents_dir: str = "backend/data/documents"
    vector_db_dir: str = "backend/data/vector_db"

    class Config:
        env_file = ".env"

settings = Settings()
