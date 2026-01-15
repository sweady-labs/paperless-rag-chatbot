from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# Optimized imports with caching
from src.rag.hybrid_vector_store import HybridVectorStore
from src.rag.query_engine import QueryEngine
from src.api.paperless_client import PaperlessClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Paperless RAG Chatbot API")

# Initialize components - models are cached
print("🚀 Initializing Paperless RAG Chatbot API...")
print("   - Using cached BGE-M3 model")
print("   - Using gemma2:2b for fast responses")
vector_store = HybridVectorStore()
query_engine = QueryEngine(vector_store, use_reranker=True)
paperless_client = PaperlessClient()
print("✅ API ready!")

class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 10  # Default 10, can be increased for analytical queries

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "name": "Paperless RAG Chatbot API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "POST /query": "Query documents using RAG",
            "GET /health": "Health check and collection info",
            "GET /stats": "Indexing statistics",
            "GET /debug/search/{query}": "Debug search results",
            "GET /debug/document/{doc_id}": "Check document content",
            "POST /reindex": "Trigger re-indexing",
            "POST /webhook/document-added": "Webhook for new documents"
        },
        "docs": "/docs (Swagger UI)",
        "redoc": "/redoc (ReDoc)"
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using optimized RAG pipeline"""
    try:
        result = query_engine.query(
            question=request.question,
            n_results=request.n_results,
            fast_rerank=True  # Immer fast reranking nutzen
        )
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        info = vector_store.get_collection_info()
        return {
            "status": "healthy",
            "collection": info,
            "indexed_documents": info['points_count'],
            "message": f"Ready! {info['points_count']} chunks indexed",
            "version": "optimized"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get indexing statistics"""
    try:
        info = vector_store.get_collection_info()
        docs = paperless_client.get_all_documents()
        return {
            "total_documents_in_paperless": len(docs),
            "indexed_chunks": info['points_count'],
            "collection_name": info['name'],
            "collection_type": info['type']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/search/{query}")
async def debug_search(query: str, n_results: int = 3):
    """Debug endpoint to see raw search results"""
    try:
        results = vector_store.search(query, n_results=n_results)
        
        debug_results = []
        for i, result in enumerate(results):
            debug_info = {
                'rank': i + 1,
                'document_id': result['metadata']['document_id'],
                'title': result['metadata']['title'],
                'chunk_index': result['metadata']['chunk_index'],
                'hybrid_score': result.get('hybrid_score', 0),
                'text_preview': result['text'][:500] + '...' if len(result['text']) > 500 else result['text'],
                'text_length': len(result['text'])
            }
            
            # Add score breakdown if available
            if 'dense_score' in result:
                debug_info['dense_score'] = result['dense_score']
            if 'sparse_score' in result:
                debug_info['sparse_score'] = result['sparse_score']
            
            debug_results.append(debug_info)
        
        return {
            'query': query,
            'results_count': len(results),
            'chunks': debug_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/document/{doc_id}")
async def debug_document(doc_id: int):
    """Check document content from Paperless"""
    try:
        doc = paperless_client.get_document(doc_id)
        return {
            'id': doc['id'],
            'title': doc.get('title', 'Untitled'),
            'has_content': bool(doc.get('content')),
            'content_length': len(doc.get('content', '')),
            'content_preview': doc.get('content', '')[:1000] + '...' if len(doc.get('content', '')) > 1000 else doc.get('content', ''),
            'created': doc.get('created'),
            'tags': doc.get('tags', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def reindex_documents():
    """Trigger re-indexing of all documents"""
    from src.indexer import DocumentIndexer
    try:
        indexer = DocumentIndexer()
        indexer.index_all_documents(clear_existing=True)
        info = vector_store.get_collection_info()
        return {
            "status": "Reindexing complete",
            "collection": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/document-added")
async def webhook_document_added(document_id: int):
    """Handle new document webhook from Paperless"""
    from src.rag.chunker import DocumentChunker
    
    try:
        doc = paperless_client.get_document(document_id)
        if not doc.get('content'):
            return {"status": "skipped", "reason": "No content"}
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        vector_store.add_chunks(chunks)
        
        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
