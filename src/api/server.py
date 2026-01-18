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
from src.monitoring.metrics import get_metrics_collector
from src.monitoring.storage import MetricsStorage
from src.monitoring.cache import get_query_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize monitoring
metrics_collector = get_metrics_collector()
metrics_storage = MetricsStorage()

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
    
    Yields JSON lines with 'type' and 'data' fields:
    - {'type': 'token', 'data': 'text chunk'}
    - {'type': 'metadata', 'data': {...}}
    """
    import json
    
    async def generate():
        try:
            async with req.app.state.query_semaphore:
                # Run generator in thread
                gen = req.app.state.query_engine.stream_query(
                    request.question,
                    request.n_results
                )
                
                for chunk in gen:
                    # Send each chunk as JSON line
                    yield json.dumps(chunk) + "\n"
                    
        except Exception as e:
            logger.exception("Streaming error")
            error_chunk = {
                'type': 'error',
                'data': str(e)
            }
            yield json.dumps(error_chunk) + "\n"
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",  # Newline-delimited JSON
        headers={"X-Accel-Buffering": "no"}
    )


@app.get("/health")
async def health(req: Request):
    """Health check endpoint."""
    try:
        # Get collection info
        collection_info = await asyncio.to_thread(
            req.app.state.vector_store.get_collection_info
        )
        
        return {
            "status": "healthy",
            "collection": {
                "name": collection_info['name'],
                "points_count": collection_info['points_count'],
                "type": f"Qdrant ({settings.OLLAMA_EMBEDDING_MODEL})"
            },
            "config": {
                "llm_model": settings.OLLAMA_MODEL,
                "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
                "use_hybrid": settings.USE_HYBRID_SEARCH,
                "dense_weight": settings.HYBRID_DENSE_WEIGHT,
                "sparse_weight": settings.HYBRID_SPARSE_WEIGHT
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
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


@app.get("/metrics/statistics")
async def get_statistics():
    """Get aggregated metrics statistics."""
    try:
        # Get in-memory stats
        mem_stats = metrics_collector.get_statistics()
        
        # Get database stats (last 24 hours)
        db_stats = metrics_storage.get_statistics(last_hours=24)
        
        return {
            "in_memory": mem_stats,
            "last_24h": db_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception("Error fetching statistics")
        return {"error": str(e)}


@app.get("/metrics/recent")
async def get_recent_queries(limit: int = 10):
    """Get recent queries with metrics."""
    try:
        # Get from database
        recent = metrics_storage.get_recent_queries(limit=limit)
        return {
            "queries": recent,
            "count": len(recent)
        }
    except Exception as e:
        logger.exception("Error fetching recent queries")
        return {"error": str(e), "queries": []}


@app.get("/metrics/slow")
async def get_slow_queries(threshold_ms: float = 5000, limit: int = 10):
    """Get slow queries above threshold."""
    try:
        slow = metrics_storage.get_slow_queries(threshold_ms=threshold_ms, limit=limit)
        return {
            "queries": slow,
            "threshold_ms": threshold_ms,
            "count": len(slow)
        }
    except Exception as e:
        logger.exception("Error fetching slow queries")
        return {"error": str(e), "queries": []}


@app.get("/metrics/live")
async def get_live_metrics():
    """Get live session metrics."""
    try:
        stats = metrics_collector.get_statistics()
        
        # Get cache stats if enabled
        cache_stats = {}
        if settings.ENABLE_CACHE:
            cache = get_query_cache()
            cache_stats = cache.get_stats()
        
        # Get recent queries from memory
        recent = metrics_collector.get_recent_queries(n=5)
        
        return {
            "statistics": stats,
            "cache": cache_stats,
            "recent_queries": recent,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception("Error fetching live metrics")
        return {"error": str(e)}


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        if not settings.ENABLE_CACHE:
            return {"enabled": False, "message": "Cache is disabled"}
        
        cache = get_query_cache()
        stats = cache.get_stats()
        top_queries = cache.get_top_queries(limit=10)
        
        return {
            "enabled": True,
            "stats": stats,
            "top_queries": [
                {"query": q, "hits": hits}
                for q, hits in top_queries
            ],
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception("Error fetching cache stats")
        return {"error": str(e)}


@app.post("/cache/clear")
async def clear_cache():
    """Clear the query cache."""
    try:
        if not settings.ENABLE_CACHE:
            return {"enabled": False, "message": "Cache is disabled"}
        
        cache = get_query_cache()
        cache.clear()
        
        return {
            "status": "success",
            "message": "Cache cleared",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.exception("Error clearing cache")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
