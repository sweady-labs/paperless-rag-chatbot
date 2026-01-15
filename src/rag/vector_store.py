from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

# Try to use native BGE-M3 for best performance
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    from langchain_community.embeddings import OllamaEmbeddings

class VectorStore:
    def __init__(self):
        self.db_path = os.getenv('VECTOR_DB_PATH', './data/vector_db')
        self.collection_name = os.getenv('COLLECTION_NAME', 'paperless_documents')
        
        # Initialize Qdrant client (local persistent storage)
        self.client = QdrantClient(path=self.db_path)
        
        # Initialize embeddings - prefer native BGE-M3 for dense+sparse+colbert
        if BGE_M3_AVAILABLE:
            print("🚀 Using native BGE-M3 with multi-functionality")
            self.model = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,  # Faster on Mac M-series
                device='cpu'
            )
            self.use_native_bge = True
        else:
            print("⚠️  FlagEmbedding not found, using Ollama (dense only)")
            self.model = OllamaEmbeddings(
                model=os.getenv('OLLAMA_EMBEDDING_MODEL', 'bge-m3'),
                base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            )
            self.use_native_bge = False
        
        # Get embedding dimension (BGE-M3 uses 1024 dimensions, supports 100+ languages)
        self.embedding_dim = 1024
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    def add_chunks(self, chunks: List[Dict]):
        """Add document chunks to vector store with BGE-M3 embeddings"""
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings based on available model
        if self.use_native_bge:
            # Use BGE-M3 native model - get dense embeddings
            # Note: For full multi-functionality (dense+sparse+colbert), use hybrid_vector_store.py
            embeddings_output = self.model.encode(
                texts,
                batch_size=12,
                max_length=8192,  # BGE-M3 supports up to 8K tokens
                return_dense=True,
                return_sparse=False,  # Simple vector store doesn't use sparse
                return_colbert_vecs=False
            )
            embeddings = embeddings_output['dense_vecs'].tolist()
        else:
            # Fallback to Ollama
            embeddings = self.model.embed_documents(texts)
        
        # Create points for Qdrant
        points = []
        for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
            point_id = f"{metadata['document_id']}_{metadata['chunk_index']}"
            # Qdrant requires integer IDs, so we'll use a hash
            point_id_int = hash(point_id) & 0x7FFFFFFF  # Ensure positive integer
            
            points.append(
                PointStruct(
                    id=point_id_int,
                    vector=embedding,
                    payload={
                        'text': text,
                        'document_id': metadata['document_id'],
                        'title': metadata['title'],
                        'created': metadata.get('created'),
                        'modified': metadata.get('modified'),
                        'correspondent': metadata.get('correspondent'),
                        'tags': metadata.get('tags', []),
                        'chunk_index': metadata['chunk_index'],
                        'total_chunks': metadata['total_chunks'],
                        'source': metadata.get('source', 'paperless-ngx')
                    }
                )
            )
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant chunks using BGE-M3 dense embeddings"""
        # Generate query embedding
        if self.use_native_bge:
            # BGE-M3: no instruction prefix needed (unlike BGE-v1.5)
            query_output = self.model.encode(
                [query],
                batch_size=1,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            query_embedding = query_output['dense_vecs'][0].tolist()
        else:
            query_embedding = self.model.embed_query(query)
        
        # Search in Qdrant using query_points (newer API)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=n_results
        ).points
        
        # Format results
        formatted_results = []
        for hit in search_result:
            formatted_results.append({
                'text': hit.payload['text'],
                'metadata': {
                    'document_id': hit.payload['document_id'],
                    'title': hit.payload['title'],
                    'created': hit.payload.get('created'),
                    'modified': hit.payload.get('modified'),
                    'correspondent': hit.payload.get('correspondent'),
                    'tags': hit.payload.get('tags', []),
                    'chunk_index': hit.payload['chunk_index'],
                    'total_chunks': hit.payload['total_chunks'],
                    'source': hit.payload.get('source', 'paperless-ngx')
                },
                'score': hit.score
            })
        
        return formatted_results
    
    def clear_collection(self):
        """Clear all data from collection"""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        info = self.client.get_collection(self.collection_name)
        return {
            'name': self.collection_name,
            'vectors_count': info.vectors_count if hasattr(info, 'vectors_count') else info.points_count,
            'points_count': info.points_count
        }
    
    def get_all_documents(self, limit: int = 10000) -> List[Dict]:
        """
        Get all documents from the vector store for hybrid search indexing.
        
        Args:
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of all documents with text and metadata
        """
        # Scroll through all points in the collection
        all_results = []
        offset = None
        
        while True:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors for BM25
            )
            
            points, next_offset = scroll_result
            
            if not points:
                break
            
            for point in points:
                all_results.append({
                    'text': point.payload['text'],
                    'metadata': {
                        'document_id': point.payload['document_id'],
                        'title': point.payload['title'],
                        'chunk_index': point.payload['chunk_index'],
                        'total_chunks': point.payload['total_chunks'],
                    }
                })
            
            offset = next_offset
            
            if next_offset is None or len(all_results) >= limit:
                break
        
        return all_results
