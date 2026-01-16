"""
FAST Vector Store using Qdrant + Ollama embeddings.

Optimized for maximum speed on Apple Silicon (M1/M2/M3/M4/M5):
- Uses Ollama's native Metal GPU acceleration
- Single embedding model (no hybrid complexity)
- LRU cache for repeated queries
- ~50ms query embedding vs ~500ms with BGE-M3

Author: Optimized for M5 GPU
"""

import logging
from typing import List, Dict, Optional
from functools import lru_cache
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_community.embeddings import OllamaEmbeddings

from src.config import settings
from src.utils.id_utils import stable_int_id

logger = logging.getLogger(__name__)

# Embedding dimension for mxbai-embed-large
MXBAI_EMBED_DIM = 1024


class FastVectorStore:
    """
    Fast vector store using Qdrant + Ollama embeddings.
    
    Key features:
    - Uses mxbai-embed-large via Ollama (native Metal support)
    - Simple dense-only search (no hybrid complexity)
    - Query embedding cache for repeated queries
    - ~10x faster than BGE-M3 hybrid search
    """
    
    def __init__(self):
        """Initialize fast vector store with Ollama embeddings."""
        self.db_path = settings.VECTOR_DB_PATH
        self.collection_name = settings.COLLECTION_NAME
        
        # Initialize Ollama embeddings (uses Metal GPU natively)
        embedding_model = settings.OLLAMA_EMBEDDING_MODEL
        logger.info(f"Initializing Ollama embeddings: {embedding_model}")
        
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            base_url=settings.OLLAMA_BASE_URL
        )
        self.dense_dim = MXBAI_EMBED_DIM
        
        # Initialize Qdrant client (local file-based)
        self.client = QdrantClient(path=self.db_path)
        
        # Ensure collection exists
        self._ensure_collection()
        
        logger.info(f"FastVectorStore ready: {self.collection_name} (dim={self.dense_dim})")
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dense_dim,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
    
    def embed_chunk(self, text: str) -> Optional[List[float]]:
        """
        Embed a single chunk of text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.warning(f"Failed to embed chunk: {e}")
            return None
    
    def add_points_batch(self, points: List[PointStruct]):
        """
        Add pre-built points to the vector store in a single batch.
        
        This is the preferred method for bulk indexing - avoids SQLite locking.
        
        Args:
            points: List of PointStruct objects with embeddings
        """
        if not points:
            return
        
        # Upsert all at once
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Added {len(points)} points to {self.collection_name}")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add document chunks to the vector store.
        
        Note: For bulk indexing, use embed_chunk() + add_points_batch() instead
        to avoid SQLite locking issues.
        
        Args:
            chunks: List of dicts with 'text' and 'metadata' keys
        """
        if not chunks:
            return
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        points = []
        for chunk in chunks:
            text = chunk['text']
            metadata = chunk['metadata']
            
            # Embed single text
            embedding = self.embed_chunk(text)
            if embedding is None:
                continue
            
            # Generate stable ID from document_id and chunk_index
            point_id = stable_int_id(f"{metadata['document_id']}_{metadata['chunk_index']}")
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'text': text,
                    'document_id': metadata['document_id'],
                    'title': metadata.get('title', 'Untitled'),
                    'created': metadata.get('created'),
                    'modified': metadata.get('modified'),
                    'correspondent': metadata.get('correspondent'),
                    'tags': metadata.get('tags', []),
                    'chunk_index': metadata['chunk_index'],
                    'total_chunks': metadata['total_chunks'],
                    'url': metadata.get('url', ''),
                }
            ))
        
        if points:
            self.add_points_batch(points)
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(query.encode()).hexdigest()
    
    @lru_cache(maxsize=500)
    def _cached_embed_query(self, query_hash: str, query: str) -> tuple:
        """Cached query embedding. Returns tuple for hashability."""
        embedding = self.embeddings.embed_query(query)
        return tuple(embedding)
    
    def search(
        self, 
        query: str, 
        n_results: int = 3,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return (default: 3)
            metadata_filter: Optional filter dict, e.g. {'correspondent': 'Amazon'}
            
        Returns:
            List of result dicts with 'text', 'metadata', 'score'
        """
        # Get cached or compute query embedding
        query_hash = self._get_query_hash(query)
        embedding_tuple = self._cached_embed_query(query_hash, query)
        query_embedding = list(embedding_tuple)
        
        # Build filter if provided
        query_filter = None
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)
        
        # Search Qdrant
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=n_results,
            query_filter=query_filter
        ).points
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'text': result.payload['text'],
                'metadata': {
                    'document_id': result.payload['document_id'],
                    'title': result.payload.get('title', 'Untitled'),
                    'created': result.payload.get('created'),
                    'modified': result.payload.get('modified'),
                    'correspondent': result.payload.get('correspondent'),
                    'tags': result.payload.get('tags', []),
                    'chunk_index': result.payload['chunk_index'],
                    'total_chunks': result.payload['total_chunks'],
                    'url': result.payload.get('url', ''),
                },
                'score': result.score
            })
        
        return formatted_results
    
    def clear_collection(self):
        """Delete and recreate collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception:
            pass
        self._ensure_collection()
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'type': 'Qdrant (mxbai-embed-large)'
            }
        except Exception:
            return {
                'name': self.collection_name,
                'points_count': 0,
                'type': 'Qdrant (mxbai-embed-large)'
            }
    
    def clear_embedding_cache(self):
        """Clear the query embedding cache."""
        self._cached_embed_query.cache_clear()
        logger.info("Cleared embedding cache")


# Helper to create PointStruct for batch indexing
def create_point(embedding: List[float], text: str, metadata: Dict) -> PointStruct:
    """Create a PointStruct for batch indexing."""
    point_id = stable_int_id(f"{metadata['document_id']}_{metadata['chunk_index']}")
    
    return PointStruct(
        id=point_id,
        vector=embedding,
        payload={
            'text': text,
            'document_id': metadata['document_id'],
            'title': metadata.get('title', 'Untitled'),
            'created': metadata.get('created'),
            'modified': metadata.get('modified'),
            'correspondent': metadata.get('correspondent'),
            'tags': metadata.get('tags', []),
            'chunk_index': metadata['chunk_index'],
            'total_chunks': metadata['total_chunks'],
            'url': metadata.get('url', ''),
        }
    )
