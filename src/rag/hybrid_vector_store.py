"""
OPTIMIZED Hybrid Vector Store using BGE-M3 with aggressive caching.

KEY OPTIMIZATIONS:
1. Model wird nur 1x geladen (singleton pattern)
2. Embeddings werden gecacht
3. Compute_score wird vermieden (zu langsam)
4. FP16 für schnellere Inferenz auf Mac
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import os
import numpy as np
from functools import lru_cache
from src.config import settings

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    logger.warning("FlagEmbedding not installed. Run: pip install FlagEmbedding")

# Global model cache - load only once!
_MODEL_CACHE = {}


def get_bge_m3_model():
    """Singleton pattern: load BGE-M3 nur einmal!"""
    if 'bge_m3' not in _MODEL_CACHE:
        logger.info("Loading BGE-M3 (one-time, will be cached)...")
        use_fp16 = settings.BGE_M3_USE_FP16
        max_length = settings.BGE_M3_MAX_LENGTH
        
        # Try to use MPS (Metal) on Mac, fallback to CPU
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
            logger.info("Using Metal Performance Shaders (GPU acceleration)")
        else:
            device = 'cpu'
            logger.info("Using CPU (no GPU acceleration available)")

        _MODEL_CACHE['bge_m3'] = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=use_fp16,
            device=device
        )
        logger.info(f"BGE-M3 loaded (fp16={use_fp16}, device={device}, max_length={max_length})")

    return _MODEL_CACHE['bge_m3']


from src.utils.id_utils import stable_int_id


class HybridVectorStore:
    """Optimized vector store mit cached BGE-M3 model."""

    def __init__(self, use_full_bge_m3: bool = True):
        self.db_path = settings.VECTOR_DB_PATH
        self.collection_name = settings.COLLECTION_NAME + '_hybrid'
        self.use_full_bge_m3 = use_full_bge_m3 and BGE_M3_AVAILABLE

        # Initialize Qdrant
        self.client = QdrantClient(path=self.db_path)

        # Use cached model!
        if self.use_full_bge_m3:
            self.model = get_bge_m3_model()
            self.dense_dim = 1024

            # Get weights from settings
            dense_w = settings.BGE_M3_DENSE_WEIGHT
            sparse_w = settings.BGE_M3_SPARSE_WEIGHT
            colbert_w = settings.BGE_M3_COLBERT_WEIGHT
            self.weights = [dense_w, sparse_w, colbert_w]
            logger.info(f"HybridVectorStore initialized (weights: dense={dense_w}, sparse={sparse_w}, colbert={colbert_w})")
        else:
            # Fallback to Ollama
            from langchain_community.embeddings import OllamaEmbeddings
            logger.warning("Using Ollama for dense-only embeddings")
            self.model = OllamaEmbeddings(
                model=settings.OLLAMA_EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
            self.dense_dim = 1024

        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if not exists."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            if self.use_full_bge_m3:
                # Hybrid: dense + sparse + colbert
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(size=self.dense_dim, distance=Distance.COSINE),
                        "colbert": VectorParams(size=self.dense_dim, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={"sparse": {}}
                )
                logger.info(f"Created hybrid collection: {self.collection_name}")
            else:
                # Dense only
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dense_dim, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {self.collection_name} (dense only)")

    def add_chunks(self, chunks: List[Dict]):
        """Add chunks with BGE-M3 embeddings (memory-optimized)."""
        if not chunks:
            return

        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")

        if self.use_full_bge_m3:
            # Memory optimization: adjust batch size based on number of chunks
            batch_size = settings.BGE_M3_BATCH_SIZE
            if len(texts) > 30:
                batch_size = max(4, batch_size // 2)  # Halve batch size for large documents
                logger.info(f"Large document detected ({len(texts)} chunks), using smaller batch size: {batch_size}")
            
            try:
                # Encode mit allen drei Modi
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    max_length=settings.BGE_M3_MAX_LENGTH,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=True
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                    logger.warning(f"GPU out of memory, retrying with batch_size=1 and clearing cache...")
                    self._clear_gpu_cache()
                    
                    # Retry with minimal batch size
                    embeddings = self.model.encode(
                        texts,
                        batch_size=1,
                        max_length=settings.BGE_M3_MAX_LENGTH,
                        return_dense=True,
                        return_sparse=True,
                        return_colbert_vecs=True
                    )
                else:
                    raise

            dense_vecs = embeddings['dense_vecs']
            sparse_vecs = embeddings['lexical_weights']
            colbert_vecs = embeddings['colbert_vecs']

            points = []
            for idx, (text, metadata) in enumerate(zip(texts, metadatas)):
                point_id = stable_int_id(f"{metadata['document_id']}_{metadata['chunk_index']}")

                # Sparse vector
                sparse_dict = sparse_vecs[idx]
                sparse_indices = list(sparse_dict.keys())
                sparse_values = list(sparse_dict.values())

                # ColBERT average
                colbert_avg = np.mean(colbert_vecs[idx], axis=0).tolist()

                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vecs[idx].tolist(),
                        "colbert": colbert_avg,
                        "sparse": {"indices": sparse_indices, "values": sparse_values}
                    },
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
                        'url': metadata.get('url', ''),  # Document URL
                    }
                )
                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Added {len(points)} chunks (dense+sparse+colbert)")

        else:
            # Fallback: dense only
            dense_vecs = [self.model.embed_query(text) for text in texts]

            points = []
            for idx, (vec, text, metadata) in enumerate(zip(dense_vecs, texts, metadatas)):
                point_id = stable_int_id(f"{metadata['document_id']}_{metadata['chunk_index']}")

                points.append(PointStruct(
                    id=point_id,
                    vector=vec,
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
                        'url': metadata.get('url', ''),  # Document URL
                    }
                ))

            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Added {len(points)} chunks (dense only)")

    def search(self, query: str, n_results: int = 10, metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        OPTIMIZED hybrid search - OHNE compute_score (zu langsam!)
        Nutzt Qdrant's native hybrid search stattdessen.

        Args:
            query: Search query
            n_results: Number of results to return
            metadata_filter: Optional filter dict, e.g. {'correspondent': 'Amazon', 'year': 2024}
        """
        if not self.use_full_bge_m3:
            # Fallback: dense only
            query_vec = self.model.embed_query(query)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=n_results
            ).points

            return [self._format_result(r, r.score) for r in results]

        # OPTIMIZED: Encode query nur 1x
        query_embeddings = self.model.encode(
            [query],
            batch_size=1,
            max_length=settings.BGE_M3_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False  # Nicht nötig für retrieval
        )

        dense_vec = query_embeddings['dense_vecs'][0]
        sparse_dict = query_embeddings['lexical_weights'][0]
        sparse_indices = list(sparse_dict.keys())
        sparse_values = list(sparse_dict.values())

        # Build Qdrant filter from metadata_filter if provided
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = None
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if conditions:
                query_filter = Filter(must=conditions)

        # Multi-vector search mit Qdrant
        # 1. Dense search - mehr initiale Ergebnisse für bessere Abdeckung
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec.tolist(),
            using="dense",
            limit=n_results * 3,  # 3x für bessere Abdeckung
            query_filter=query_filter
        ).points

        # 2. Sparse search - besonders wichtig für Keywords (Amazon, Monate, etc.)
        try:
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query={"indices": sparse_indices, "values": sparse_values},
                using="sparse",
                limit=n_results * 3,  # 3x für bessere Keyword-Abdeckung
                query_filter=query_filter
            ).points
        except Exception:
            sparse_results = []

        # 3. Combine results mit weighted fusion
        combined_scores = {}

        for r in dense_results:
            pid = r.id
            combined_scores[pid] = {
                'dense_score': r.score,
                'sparse_score': 0.0,
                'result': r
            }

        for r in sparse_results:
            pid = r.id
            if pid in combined_scores:
                combined_scores[pid]['sparse_score'] = r.score
            else:
                combined_scores[pid] = {
                    'dense_score': 0.0,
                    'sparse_score': r.score,
                    'result': r
                }

        # Calculate hybrid scores
        ranked_results = []
        for pid, scores in combined_scores.items():
            hybrid_score = (
                self.weights[0] * scores['dense_score'] +
                self.weights[1] * scores['sparse_score']
            )
            ranked_results.append({
                'result': scores['result'],
                'hybrid_score': hybrid_score,
                'dense_score': scores['dense_score'],
                'sparse_score': scores['sparse_score']
            })

        # Sort by hybrid score
        ranked_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

        # Format top results
        final_results = []
        for item in ranked_results[:n_results]:
            r = item['result']
            formatted = self._format_result(r, item['hybrid_score'])
            formatted['dense_score'] = item['dense_score']
            formatted['sparse_score'] = item['sparse_score']
            final_results.append(formatted)

        return final_results

    def _format_result(self, result, score: float) -> Dict:
        """Format Qdrant result to standard dict."""
        return {
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
                'url': result.payload.get('url', ''),  # Document URL
            },
            'hybrid_score': score
        }

    def clear_collection(self):
        """Delete and recreate collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception:
            pass
        self._ensure_collection()

    def get_collection_info(self) -> Dict:
        """Get collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            'name': self.collection_name,
            'points_count': info.points_count,
            'type': 'hybrid (dense+sparse+colbert)' if self.use_full_bge_m3 else 'dense only'
        }
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to free memory."""
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("Cleared MPS cache")
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")
            import gc
            gc.collect()
        except Exception as e:
            logger.debug(f"Could not clear GPU cache: {e}")
