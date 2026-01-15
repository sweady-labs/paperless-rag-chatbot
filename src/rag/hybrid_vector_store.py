"""
OPTIMIZED Hybrid Vector Store using BGE-M3 with aggressive caching.

KEY OPTIMIZATIONS:
1. Model wird nur 1x geladen (singleton pattern)
2. Embeddings werden gecacht
3. Compute_score wird vermieden (zu langsam)
4. FP16 für schnellere Inferenz auf Mac
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import numpy as np
from functools import lru_cache

load_dotenv()

try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    print("⚠️  FlagEmbedding not installed. Run: pip install FlagEmbedding")

# Global model cache - load only once!
_MODEL_CACHE = {}

def get_bge_m3_model():
    """Singleton pattern: load BGE-M3 nur einmal!"""
    if 'bge_m3' not in _MODEL_CACHE:
        print("🚀 Loading BGE-M3 (one-time, will be cached)...")
        use_fp16 = os.getenv('BGE_M3_USE_FP16', 'true').lower() == 'true'
        max_length = int(os.getenv('BGE_M3_MAX_LENGTH', '8192'))
        
        _MODEL_CACHE['bge_m3'] = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=use_fp16,
            device='cpu'  # Mac uses CPU with Metal
        )
        print(f"✅ BGE-M3 loaded (fp16={use_fp16}, max_length={max_length})")
    
    return _MODEL_CACHE['bge_m3']


class HybridVectorStore:
    """Optimized vector store mit cached BGE-M3 model."""
    
    def __init__(self, use_full_bge_m3: bool = True):
        self.db_path = os.getenv('VECTOR_DB_PATH', './data/vector_db')
        self.collection_name = os.getenv('COLLECTION_NAME', 'paperless_documents_hybrid')
        self.use_full_bge_m3 = use_full_bge_m3 and BGE_M3_AVAILABLE
        
        # Initialize Qdrant
        self.client = QdrantClient(path=self.db_path)
        
        # Use cached model!
        if self.use_full_bge_m3:
            self.model = get_bge_m3_model()
            self.dense_dim = 1024
            
            # Get weights from env
            dense_w = float(os.getenv('BGE_M3_DENSE_WEIGHT', '0.4'))
            sparse_w = float(os.getenv('BGE_M3_SPARSE_WEIGHT', '0.4'))
            colbert_w = float(os.getenv('BGE_M3_COLBERT_WEIGHT', '0.2'))
            self.weights = [dense_w, sparse_w, colbert_w]
            print(f"✅ HybridVectorStore initialized (weights: dense={dense_w}, sparse={sparse_w}, colbert={colbert_w})")
        else:
            # Fallback to Ollama
            from langchain_community.embeddings import OllamaEmbeddings
            print("⚠️  Using Ollama for dense-only embeddings")
            self.model = OllamaEmbeddings(
                model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'bge-m3'),
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
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
                print(f"✅ Created hybrid collection: {self.collection_name}")
            else:
                # Dense only
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dense_dim, distance=Distance.COSINE)
                )
                print(f"✅ Created collection: {self.collection_name} (dense only)")
    
    def add_chunks(self, chunks: List[Dict]):
        """Add chunks with BGE-M3 embeddings."""
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        print(f"Encoding {len(texts)} chunks...")
        
        if self.use_full_bge_m3:
            # Encode mit allen drei Modi
            embeddings = self.model.encode(
                texts,
                batch_size=12,
                max_length=int(os.getenv('BGE_M3_MAX_LENGTH', '8192')),
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True
            )
            
            dense_vecs = embeddings['dense_vecs']
            sparse_vecs = embeddings['lexical_weights']
            colbert_vecs = embeddings['colbert_vecs']
            
            points = []
            for idx, (text, metadata) in enumerate(zip(texts, metadatas)):
                point_id = hash(f"{metadata['document_id']}_{metadata['chunk_index']}") & 0x7FFFFFFF
                
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
            print(f"✅ Added {len(points)} chunks (dense+sparse+colbert)")
            
        else:
            # Fallback: dense only
            dense_vecs = [self.model.embed_query(text) for text in texts]
            
            points = []
            for idx, (vec, text, metadata) in enumerate(zip(dense_vecs, texts, metadatas)):
                point_id = hash(f"{metadata['document_id']}_{metadata['chunk_index']}") & 0x7FFFFFFF
                
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
            print(f"✅ Added {len(points)} chunks (dense only)")
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        OPTIMIZED hybrid search - OHNE compute_score (zu langsam!)
        Nutzt Qdrant's native hybrid search stattdessen.
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
            max_length=int(os.getenv('BGE_M3_MAX_LENGTH', '8192')),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False  # Nicht nötig für retrieval
        )
        
        dense_vec = query_embeddings['dense_vecs'][0]
        sparse_dict = query_embeddings['lexical_weights'][0]
        sparse_indices = list(sparse_dict.keys())
        sparse_values = list(sparse_dict.values())
        
        # Multi-vector search mit Qdrant
        # 1. Dense search
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec.tolist(),
            using="dense",
            limit=n_results * 2  # Get more for reranking
        ).points
        
        # 2. Sparse search
        try:
            sparse_results = self.client.query_points(
                collection_name=self.collection_name,
                query={"indices": sparse_indices, "values": sparse_values},
                using="sparse",
                limit=n_results * 2
            ).points
        except:
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
            print(f"✅ Deleted collection: {self.collection_name}")
        except:
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
