"""
Configuration settings for Paperless RAG Chatbot.
Optimized for Apple Silicon (M5) with fast Ollama embeddings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # Paperless-NGX
    # ==========================================================================
    PAPERLESS_URL: str = Field(default='http://localhost:8000')
    PAPERLESS_TOKEN: Optional[str] = Field(default=None)

    # ==========================================================================
    # Ollama Configuration
    # ==========================================================================
    OLLAMA_BASE_URL: str = Field(default='http://localhost:11434')
    OLLAMA_MODEL: str = Field(default='qwen2.5:3b')
    OLLAMA_EMBEDDING_MODEL: str = Field(default='nomic-embed-text')

    # ==========================================================================
    # LLM Generation Settings
    # ==========================================================================
    LLM_MAX_TOKENS: int = Field(default=200)
    LLM_NUM_CTX: int = Field(default=2048)
    LLM_TEMPERATURE: float = Field(default=0.0)
    ENABLE_STREAMING: bool = Field(default=True)

    # ==========================================================================
    # Chunking Configuration
    # ==========================================================================
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=50)
    MAX_CHUNK_SIZE: int = Field(default=2048)

    # ==========================================================================
    # Vector Database
    # ==========================================================================
    VECTOR_DB_PATH: str = Field(default='./data/vector_db')
    COLLECTION_NAME: str = Field(default='paperless_fast')

    # ==========================================================================
    # Retrieval Settings
    # ==========================================================================
    TOP_K: int = Field(default=3)
    RERANK_ENABLED: bool = Field(default=False)
    RERANK_TOP_K: int = Field(default=10)

    # ==========================================================================
    # Performance
    # ==========================================================================
    MAX_CONCURRENT_QUERIES: int = Field(default=4)
    AUTO_INDEX: bool = Field(default=False)

    # ==========================================================================
    # Legacy BGE-M3 Settings (kept for backwards compatibility)
    # ==========================================================================
    BGE_M3_USE_FP16: bool = Field(default=True)
    BGE_M3_MAX_LENGTH: int = Field(default=8192)
    BGE_M3_DENSE_WEIGHT: float = Field(default=0.4)
    BGE_M3_SPARSE_WEIGHT: float = Field(default=0.4)
    BGE_M3_COLBERT_WEIGHT: float = Field(default=0.2)
    BGE_M3_BATCH_SIZE: int = Field(default=8)
    BGE_M3_MAX_CHUNKS_PER_BATCH: int = Field(default=50)
    BGE_RERANKER_MODEL: str = Field(default='BAAI/bge-reranker-v2-m3')

    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'extra': 'ignore'  # Ignore extra fields in .env
    }


settings = Settings()
