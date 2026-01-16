from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import logging

from src.rag.hybrid_vector_store import HybridVectorStore
from src.rag.query_engine import QueryEngine
from src.api.paperless_client import PaperlessClient
from src.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(title="Paperless RAG Chatbot API")

# Components will be initialized on startup and stored in app.state

class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 10  # Default 10, can be increased for analytical queries

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


@app.on_event("startup")
async def startup_event():
    """Initialize heavy components once on application startup and attach to app.state."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Paperless RAG Chatbot API (startup)")

    app.state.vector_store = HybridVectorStore()
    app.state.query_engine = QueryEngine(app.state.vector_store, use_reranker=True)
    app.state.paperless_client = PaperlessClient()

    # concurrency semaphore to limit concurrent LLM/embedding requests
    app.state.query_semaphore = asyncio.Semaphore(int(settings.MAX_CONCURRENT_QUERIES))
    logger.info("API ready")


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
async def query_documents(request: QueryRequest, req: Request):
    """Query documents using optimized RAG pipeline"""
    try:
        async with req.app.state.query_semaphore:
            result = await asyncio.to_thread(
                req.app.state.query_engine.query,
                request.question,
                request.n_results,
                True  # fast_rerank
            )

        return QueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    except Exception as e:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check(req: Request):
    """Health check endpoint"""
    try:
        info = await asyncio.to_thread(req.app.state.vector_store.get_collection_info)
        return {
            "status": "healthy",
            "collection": info,
            "indexed_documents": info['points_count'],
            "message": f"Ready! {info['points_count']} chunks indexed",
            "version": "optimized"
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/stats")
async def get_stats(req: Request):
    """Get indexing statistics"""
    try:
        info, docs = await asyncio.gather(
            asyncio.to_thread(req.app.state.vector_store.get_collection_info),
            asyncio.to_thread(req.app.state.paperless_client.get_all_documents)
        )
        return {
            "total_documents_in_paperless": len(docs),
            "indexed_chunks": info['points_count'],
            "collection_name": info['name'],
            "collection_type": info['type']
        }
    except Exception as e:
        logger.exception("Failed to get stats")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/search/{query}")
async def debug_search(query: str, n_results: int = 3, req: Request = None):
    """Debug endpoint to see raw search results"""
    try:
        results = await asyncio.to_thread(req.app.state.vector_store.search, query, n_results)

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
        logger.exception("Debug search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/document/{doc_id}")
async def debug_document(doc_id: int, req: Request):
    """Check document content from Paperless"""
    try:
        doc = await asyncio.to_thread(req.app.state.paperless_client.get_document, doc_id)
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
        logger.exception("Failed to fetch document")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex_documents(background_tasks: BackgroundTasks, req: Request):
    """Trigger re-indexing of all documents"""
    from src.indexer import DocumentIndexer
    try:
        # Run indexing in background to avoid blocking API
        def run_indexer():
            indexer = DocumentIndexer()
            indexer.index_all_documents(clear_existing=True)

        background_tasks.add_task(run_indexer)

        info = await asyncio.to_thread(req.app.state.vector_store.get_collection_info)
        return {
            "status": "Reindexing started",
            "collection": info
        }
    except Exception as e:
        logger.exception("Failed to start reindexing")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/document-added")
async def webhook_document_added(document_id: int, req: Request):
    """Handle new document webhook from Paperless"""
    from src.rag.chunker import DocumentChunker

    try:
        doc = await asyncio.to_thread(req.app.state.paperless_client.get_document, document_id)
        if not doc.get('content'):
            return {"status": "skipped", "reason": "No content"}

        chunker = DocumentChunker()
        chunks = await asyncio.to_thread(chunker.chunk_document, doc)
        await asyncio.to_thread(req.app.state.vector_store.add_chunks, chunks)

        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        logger.exception("Failed to index webhook document")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
