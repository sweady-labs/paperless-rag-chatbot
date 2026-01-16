from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Paperless
    PAPERLESS_URL: str = Field('http://localhost:8000', env='PAPERLESS_URL')
    PAPERLESS_TOKEN: str | None = Field(None, env='PAPERLESS_TOKEN')

    # Ollama
    OLLAMA_BASE_URL: str = Field('http://localhost:11434', env='OLLAMA_BASE_URL')
    OLLAMA_MODEL: str = Field('gemma2:2b', env='OLLAMA_MODEL')
    OLLAMA_EMBEDDING_MODEL: str = Field('bge-m3', env='OLLAMA_EMBEDDING_MODEL')

    # BGE-M3
    BGE_M3_USE_FP16: bool = Field(True, env='BGE_M3_USE_FP16')
    BGE_M3_MAX_LENGTH: int = Field(8192, env='BGE_M3_MAX_LENGTH')
    BGE_M3_DENSE_WEIGHT: float = Field(0.4, env='BGE_M3_DENSE_WEIGHT')
    BGE_M3_SPARSE_WEIGHT: float = Field(0.4, env='BGE_M3_SPARSE_WEIGHT')
    BGE_M3_COLBERT_WEIGHT: float = Field(0.2, env='BGE_M3_COLBERT_WEIGHT')

    # Chunking
    CHUNK_SIZE: int = Field(1000, env='CHUNK_SIZE')
    CHUNK_OVERLAP: int = Field(200, env='CHUNK_OVERLAP')
    MAX_CHUNK_SIZE: int = Field(8000, env='MAX_CHUNK_SIZE')

    # Vector DB
    VECTOR_DB_PATH: str = Field('./data/vector_db', env='VECTOR_DB_PATH')
    COLLECTION_NAME: str = Field('paperless_documents', env='COLLECTION_NAME')

    # Reranker
    BGE_RERANKER_MODEL: str = Field('BAAI/bge-reranker-v2-m3', env='BGE_RERANKER_MODEL')

    # Misc
    AUTO_INDEX: bool = Field(False, env='AUTO_INDEX')
    RERANK_TOP_K: int = Field(10, env='RERANK_TOP_K')

    # Concurrency
    MAX_CONCURRENT_QUERIES: int = Field(4, env='MAX_CONCURRENT_QUERIES')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()
