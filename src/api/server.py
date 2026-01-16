"""
Fast API Server for Paperless RAG Chatbot.

Optimized for sub-2-second response times on Apple Silicon.
"""

import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from src.config import settings
from src.rag.fast_vector_store import FastVectorStore
from src.rag.fast_query_engine import FastQueryEngine
from src.api.paperless_client import PaperlessClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Paperless RAG Chatbot API",
    description="Fast RAG-based document Q&A for Paperless-NGX",
    version="3.0-fast"
)


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    n_results: Optional[int] = None  # Uses settings.TOP_K if not specified
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[dict]
    latency_ms: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("=" * 50)
    logger.info("Starting Paperless RAG Chatbot API (Fast Mode)")
    logger.info("=" * 50)
    
    # Initialize fast components
    app.state.vector_store = FastVectorStore()
    app.state.query_engine = FastQueryEngine(app.state.vector_store)
    app.state.paperless_client = PaperlessClient()
    
    # Concurrency semaphore
    app.state.query_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUERIES)
    
    logger.info(f"LLM Model: {settings.OLLAMA_MODEL}")
    logger.info(f"Embedding Model: {settings.OLLAMA_EMBEDDING_MODEL}")
    logger.info(f"Collection: {settings.COLLECTION_NAME}")
    logger.info("API Ready!")


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Paperless RAG Chatbot API",
        "version": "3.0-fast",
        "status": "running",
        "config": {
            "llm_model": settings.OLLAMA_MODEL,
            "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
            "top_k": settings.TOP_K,
            "streaming": settings.ENABLE_STREAMING
        },
        "endpoints": {
            "POST /query": "Query documents",
            "GET /health": "Health check",
            "GET /stats": "Collection statistics",
            "POST /reindex": "Trigger re-indexing",
            "GET /debug/search/{query}": "Debug search"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, req: Request):
    """
    Query documents using fast RAG pipeline.
    
    Expected latency: 1-2 seconds on Apple Silicon.
    """
    start_time = time.time()
    
    try:
        async with req.app.state.query_semaphore:
            result = await asyncio.to_thread(
                req.app.state.query_engine.query,
                request.question,
                request.n_results
            )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result['answer'],
            sources=result['sources'],
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest, req: Request):
    """
    Stream query response for perceived speed.
    
    Returns tokens as they're generated.
    """
    async def generate():
        try:
            async with req.app.state.query_semaphore:
                # Run generator in thread
                gen = req.app.state.query_engine.stream_query(
                    request.question,
                    request.n_results
                )
                
                for token in gen:
                    yield token
                    
        except Exception as e:
            logger.exception("Streaming error")
            yield f"\n\nError: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"}
    )


@app.get("/health")
async def health_check(req: Request):
    """Health check with collection info."""
    try:
        info = await asyncio.to_thread(
            req.app.state.vector_store.get_collection_info
        )
        return {
            "status": "healthy",
            "collection": info,
            "config": {
                "llm_model": settings.OLLAMA_MODEL,
                "embedding_model": settings.OLLAMA_EMBEDDING_MODEL
            }
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/stats")
async def get_stats(req: Request):
    """Get indexing statistics."""
    try:
        info = await asyncio.to_thread(
            req.app.state.vector_store.get_collection_info
        )
        docs = await asyncio.to_thread(
            req.app.state.paperless_client.get_all_documents
        )
        
        return {
            "paperless_documents": len(docs),
            "indexed_chunks": info['points_count'],
            "collection": info['name'],
            "type": info['type']
        }
    except Exception as e:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/search/{query}")
async def debug_search(query: str, n_results: int = 5, req: Request = None):
    """Debug endpoint to see raw search results."""
    try:
        results = await asyncio.to_thread(
            req.app.state.vector_store.search,
            query,
            n_results
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "document_id": r['metadata']['document_id'],
                    "title": r['metadata']['title'],
                    "score": round(r.get('score', 0), 4),
                    "text_preview": r['text'][:300] + "..." if len(r['text']) > 300 else r['text']
                }
                for i, r in enumerate(results)
            ]
        }
    except Exception as e:
        logger.exception("Debug search error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex_documents(background_tasks: BackgroundTasks):
    """Trigger background re-indexing."""
    from src.indexer import FastIndexer
    
    def run_indexer():
        indexer = FastIndexer()
        indexer.index_all_documents(clear_existing=True)
    
    background_tasks.add_task(run_indexer)
    
    return {
        "status": "Re-indexing started in background",
        "config": {
            "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
            "chunk_size": settings.CHUNK_SIZE
        }
    }


@app.post("/webhook/document-added")
async def webhook_document_added(document_id: int, req: Request):
    """Handle new document webhook from Paperless."""
    from src.rag.chunker import DocumentChunker
    
    try:
        doc = await asyncio.to_thread(
            req.app.state.paperless_client.get_document,
            document_id
        )
        
        if not doc.get('content'):
            return {"status": "skipped", "reason": "No content"}
        
        chunker = DocumentChunker()
        chunks = await asyncio.to_thread(chunker.chunk_document, doc)
        await asyncio.to_thread(
            req.app.state.vector_store.add_chunks,
            chunks
        )
        
        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        logger.exception("Webhook error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
