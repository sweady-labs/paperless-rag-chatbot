"""
Hybrid Retriever combining Dense (Vector) + Sparse (BM25) search.

This implements the GPT-4 recommendation for better retrieval quality.
Combines semantic search with keyword matching for best results.
"""

import logging
from typing import List, Dict
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Dense vector search (semantic similarity via embeddings)
    - Sparse BM25 search (keyword matching for exact terms)
    
    This addresses the main weakness of pure vector search: missing exact matches
    like names ("mavi", "luke"), invoice numbers, dates, etc.
    """
    
    def __init__(self, vector_store):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: FastVectorStore instance
        """
        self.vector_store = vector_store
        # Reuse existing client to avoid file locking issues
        self.client = vector_store.client
        self.collection_name = vector_store.collection_name
        
        # Build BM25 index
        logger.info("Building BM25 index for keyword search...")
        self._build_bm25_index()
        logger.info(f"BM25 index built with {len(self.documents)} documents")
    
    def _build_bm25_index(self):
        """Build BM25 index from all documents in Qdrant."""
        # Fetch all documents from Qdrant
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=100000,  # Get all points
            with_payload=True,
            with_vectors=False
        )
        
        # Store documents and their metadata
        self.documents = []
        self.doc_metadata = []
        
        for point in points:
            text = point.payload.get('text', '')
            # Tokenize for BM25 (simple whitespace + lowercase)
            tokens = text.lower().split()
            
            self.documents.append(tokens)
            self.doc_metadata.append({
                'point_id': point.id,
                'text': text,
                'metadata': {
                    'document_id': point.payload.get('document_id'),
                    'title': point.payload.get('title', 'Untitled'),
                    'created': point.payload.get('created'),
                    'modified': point.payload.get('modified'),
                    'correspondent': point.payload.get('correspondent'),
                    'tags': point.payload.get('tags', []),
                    'chunk_index': point.payload.get('chunk_index', 0),
                    'total_chunks': point.payload.get('total_chunks', 1),
                    'url': point.payload.get('url', ''),
                }
            })
        
        # Build BM25 index
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
        else:
            self.bm25 = None
            logger.warning("No documents found for BM25 index")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        metadata_filter: Dict = None
    ) -> List[Dict]:
        """
        Hybrid search combining dense vector + sparse BM25.
        
        Args:
            query: Search query
            n_results: Number of results to return
            dense_weight: Weight for vector search (0.0-1.0)
            sparse_weight: Weight for BM25 search (0.0-1.0)
            metadata_filter: Optional metadata filter
            
        Returns:
            List of results with combined scores
        """
        # 1. Dense vector search (semantic)
        dense_results = self.vector_store.search(
            query,
            n_results=n_results * 2,  # Get more candidates
            metadata_filter=metadata_filter
        )
        
        # 2. Sparse BM25 search (keyword)
        sparse_results = []
        if self.bm25:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get top N by BM25 score
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:n_results * 2]
            
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only include non-zero scores
                    doc = self.doc_metadata[idx].copy()
                    doc['bm25_score'] = float(bm25_scores[idx])
                    sparse_results.append(doc)
        
        # 3. Combine and rerank
        combined = self._combine_results(
            dense_results,
            sparse_results,
            dense_weight,
            sparse_weight
        )
        
        # 4. Return top N
        return combined[:n_results]
    
    def _combine_results(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Dict]:
        """
        Combine dense and sparse results using weighted scoring.
        
        Uses Reciprocal Rank Fusion (RRF) for combining rankings.
        """
        # Normalize scores to 0-1 range
        def normalize_scores(results, score_key):
            if not results:
                return []
            scores = [r.get(score_key, 0) for r in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            for r in results:
                raw_score = r.get(score_key, 0)
                r[f'{score_key}_norm'] = (raw_score - min_score) / score_range if score_range > 0 else 0
            return results
        
        # Normalize both result sets
        dense_results = normalize_scores(dense_results, 'score')
        sparse_results = normalize_scores(sparse_results, 'bm25_score')
        
        # Merge by point_id / text (use text as fallback)
        merged = {}
        
        # Add dense results
        for r in dense_results:
            key = r.get('text', '')[:100]  # Use text snippet as key
            merged[key] = {
                'text': r.get('text'),
                'metadata': r.get('metadata'),
                'dense_score': r.get('score', 0),
                'dense_score_norm': r.get('score_norm', 0),
                'bm25_score': 0,
                'bm25_score_norm': 0,
            }
        
        # Add/merge sparse results
        for r in sparse_results:
            key = r.get('text', '')[:100]
            if key in merged:
                merged[key]['bm25_score'] = r.get('bm25_score', 0)
                merged[key]['bm25_score_norm'] = r.get('bm25_score_norm', 0)
            else:
                merged[key] = {
                    'text': r.get('text'),
                    'metadata': r.get('metadata'),
                    'dense_score': 0,
                    'dense_score_norm': 0,
                    'bm25_score': r.get('bm25_score', 0),
                    'bm25_score_norm': r.get('bm25_score_norm', 0),
                }
        
        # Calculate hybrid scores
        results = []
        for key, r in merged.items():
            hybrid_score = (
                dense_weight * r['dense_score_norm'] +
                sparse_weight * r['bm25_score_norm']
            )
            
            results.append({
                'text': r['text'],
                'metadata': r['metadata'],
                'score': hybrid_score,  # Combined score
                'dense_score': r['dense_score'],
                'bm25_score': r['bm25_score'],
            })
        
        # Sort by hybrid score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def rebuild_index(self):
        """Rebuild BM25 index (call after indexing new documents)."""
        self._build_bm25_index()
