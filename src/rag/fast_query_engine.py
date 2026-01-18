"""
Fast Query Engine optimized for Apple Silicon.

Key optimizations:
- Uses qwen2.5:3b for fast inference (~35-50 tok/s on M5)
- Smaller context window (2048 tokens)
- Shorter max output (200 tokens)
- Hybrid search (dense vector + sparse BM25) for better recall
- Optional streaming for perceived speed

Author: Optimized for M5 GPU
"""

import logging
import re
import time
from typing import List, Dict, Optional, Generator

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from src.config import settings
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.synonyms import expand_query_with_synonyms, should_expand_query
from src.monitoring.logger import QueryLogger
from src.monitoring.metrics import QueryMetrics, get_metrics_collector
from src.monitoring.storage import MetricsStorage
from src.monitoring.cache import get_query_cache

logger = logging.getLogger(__name__)

# Initialize monitoring
metrics_collector = get_metrics_collector()
metrics_storage = MetricsStorage()

# Initialize cache (lazy loaded)
_query_cache = None


def get_cache():
    """Get query cache instance (lazy initialization)."""
    global _query_cache
    if _query_cache is None and settings.ENABLE_CACHE:
        _query_cache = get_query_cache()
    return _query_cache


class FastQueryEngine:
    """
    Optimized query engine for fast RAG responses.
    
    Designed for sub-2-second end-to-end latency on Apple Silicon.
    Now with hybrid search (vector + BM25) for better quality.
    """
    
    def __init__(self, vector_store, use_hybrid: bool = None):
        """
        Initialize the fast query engine.
        
        Args:
            vector_store: FastVectorStore instance for retrieval
            use_hybrid: Use hybrid search (vector + BM25) instead of pure vector.
                       If None, reads from settings.USE_HYBRID_SEARCH
        """
        self.vector_store = vector_store
        
        # Use settings if not explicitly specified
        if use_hybrid is None:
            use_hybrid = settings.USE_HYBRID_SEARCH
        
        self.use_hybrid = use_hybrid
        
        # Initialize hybrid retriever if enabled
        if use_hybrid:
            logger.info("Initializing hybrid retriever (vector + BM25)...")
            self.retriever = HybridRetriever(vector_store)
            logger.info("Hybrid retriever ready")
        else:
            self.retriever = None
        
        # Initialize Ollama LLM with optimized settings
        model = settings.OLLAMA_MODEL
        self.llm = Ollama(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            num_predict=settings.LLM_MAX_TOKENS,
            num_ctx=settings.LLM_NUM_CTX,
            timeout=30,  # 30s timeout is plenty for fast models
            top_k=10,
            top_p=0.9,
        )
        
        logger.info(f"FastQueryEngine initialized with {model}")
        logger.info(f"  - max_tokens: {settings.LLM_MAX_TOKENS}")
        logger.info(f"  - num_ctx: {settings.LLM_NUM_CTX}")
        logger.info(f"  - streaming: {settings.ENABLE_STREAMING}")
        
        # Compact prompt template for speed (fewer tokens = faster)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Answer based ONLY on the documents below. If the answer isn't in the documents, say so.

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
        )
        
        # German prompt for German queries - Strict but helpful
        self.prompt_template_de = PromptTemplate(
            input_variables=["context", "question"],
            template="""Du bist ein präziser Assistent für Dokumentensuche. Beantworte die Frage basierend auf den Dokumenten unten.

WICHTIGE REGELN:
1. Nutze NUR Informationen aus den bereitgestellten Dokumenten
2. Wenn ein Dokument NICHT zur Frage passt, ignoriere es
3. Wenn die Antwort WIRKLICH nicht in den Dokumenten steht, sage: "Die Antwort ist in den Dokumenten nicht enthalten."
4. Sei präzise und direkt - antworte in 1-3 Sätzen
5. Extrahiere relevante Informationen auch aus OCR-Text mit Fehlern

DOKUMENTE:
{context}

FRAGE: {question}

ANTWORT:"""
        )
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on common words.
        Returns 'de' for German, 'en' for English.
        Default to German for ambiguous cases (primary use case).
        """
        # Extended German indicators including common query words
        german_indicators = [
            'der', 'die', 'das', 'und', 'ist', 'von', 'mit', 'für', 
            'was', 'wie', 'wo', 'wann', 'welche', 'welcher', 'welches',
            'wurde', 'wurden', 'hatte', 'haben', 'bei', 'zu', 'im', 'in',
            'ein', 'eine', 'einen', 'dem', 'den', 'des', 'alle', 'mir',
            'aus', 'auf', 'über', 'unter', 'nach', 'vor', 'zwischen',
            'seit', 'bis', 'durch', 'gegen', 'ohne', 'um'
        ]
        
        # German-specific words/compounds (even without spaces)
        german_compounds = [
            'rechnung', 'steuer', 'bescheid', 'behandlung', 'arzt',
            'versicherung', 'vertrag', 'urkunde', 'geboren', 'einkommens',
            'grund', 'erwerb', 'volkswagen'
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Check for German indicator words
        german_count = sum(1 for word in words if word in german_indicators)
        
        # Check for German compounds (partial matches)
        compound_count = sum(1 for compound in german_compounds if compound in text_lower)
        
        # Total German signals
        total_german = german_count + compound_count
        
        # If 1+ German signals found, use German (lowered threshold from 2)
        # Default to German for ambiguous single-word queries
        if total_german >= 1 or len(words) == 1:
            return 'de'
        
        return 'en'
    
    def _extract_metadata_filter(self, question: str) -> Optional[Dict]:
        """Extract metadata filters from question."""
        filter_dict = {}
        
        # Extract correspondent names
        correspondent_patterns = [
            r'bei\s+(\w+)',
            r'von\s+(\w+)',
            r'at\s+(\w+)',
            r'from\s+(\w+)',
        ]
        for pattern in correspondent_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                correspondent = match.group(1).capitalize()
                if len(correspondent) > 2:
                    filter_dict['correspondent'] = correspondent
                    break
        
        return filter_dict if filter_dict else None
    
    def query(self, question: str, n_results: int = None, min_score: float = 0.50) -> Dict:
        """
        Answer a question using fast RAG pipeline with comprehensive monitoring.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve (default from settings)
            min_score: Minimum similarity score to include (default: 0.50 for better precision)
            
        Returns:
            Dict with 'answer', 'sources', 'context', 'method', 'metrics'
        """
        # Start timing
        query_start_time = time.time()
        
        # Try cache first
        cache = get_cache()
        if cache:
            cached_result = cache.get(question)
            if cached_result is not None:
                # Cache hit! Return instantly
                logger.info(f"Cache HIT: '{question}' - Instant response!")
                
                # Add cache hit to result metrics
                if 'metrics' in cached_result:
                    cached_result['metrics']['cache_hit'] = True
                    cached_result['metrics']['cache_latency_ms'] = round((time.time() - query_start_time) * 1000, 2)
                
                return cached_result
        
        # Initialize monitoring
        with QueryLogger(question) as qlog:
            try:
                if n_results is None:
                    n_results = settings.TOP_K
                
                # Detect if analytical query (needs more context)
                analytical_keywords = [
                    'alle', 'gesamt', 'übersicht', 'liste', 'tabelle',
                    'all', 'total', 'overview', 'list', 'table', 'every'
                ]
                is_analytical = any(kw in question.lower() for kw in analytical_keywords)
                if is_analytical:
                    n_results = min(n_results * 3, 10)  # More results but capped
                
                # Extract metadata filter
                metadata_filter = self._extract_metadata_filter(question)
                
                # Apply synonym expansion if enabled
                expanded_question = question
                if settings.ENABLE_SYNONYM_EXPANSION and should_expand_query(question):
                    expanded_question = expand_query_with_synonyms(question)
                    logger.info(f"Synonym expansion: '{question}' → '{expanded_question}'")
                    qlog.log_stage('synonym_expansion', duration_ms=0, details={
                        'original': question,
                        'expanded': expanded_question
                    })
                
                # 1. Retrieve relevant chunks (hybrid or vector-only)
                search_start = time.time()
                
                if self.use_hybrid and self.retriever:
                    # Hybrid search (vector + BM25) with configured weights
                    results = self.retriever.search(
                        expanded_question,
                        n_results=n_results,
                        dense_weight=settings.HYBRID_DENSE_WEIGHT,
                        sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
                        metadata_filter=metadata_filter
                    )
                else:
                    # Pure vector search (fallback)
                    results = self.vector_store.search(
                        expanded_question, 
                        n_results=n_results,
                        metadata_filter=metadata_filter
                    )
                
                search_duration = (time.time() - search_start) * 1000
                
                # Store pre-filter count
                num_candidates = len(results)
                
                # Dynamic min-score threshold based on BM25 scores - More conservative
                # High BM25 = strong keyword match = lower threshold acceptable
                # Low BM25 = semantic only = stricter threshold
                if results and self.use_hybrid:
                    top_bm25 = max([r.get('bm25_score', 0) for r in results])
                    
                    if top_bm25 > 15.0:
                        # Very strong keyword match - slightly lower threshold
                        dynamic_threshold = max(min_score * 0.85, 0.45)  # But not below 0.45
                        logger.debug(f"Dynamic threshold: {dynamic_threshold:.2f} (high BM25: {top_bm25:.2f})")
                    elif top_bm25 < 2.0:
                        # Weak keyword match - rely on semantic, be very strict
                        dynamic_threshold = min(min_score * 1.3, 0.75)
                        logger.debug(f"Dynamic threshold: {dynamic_threshold:.2f} (low BM25: {top_bm25:.2f})")
                    else:
                        # Medium keyword match - use default
                        dynamic_threshold = min_score
                        logger.debug(f"Dynamic threshold: {min_score} (medium BM25: {top_bm25:.2f})")
                    
                    min_score = dynamic_threshold
                
                # Filter by minimum score
                if min_score > 0:
                    results = [r for r in results if r.get('score', 0) >= min_score]
                
                num_filtered = len(results)
                
                # Log search results
                if results:
                    top_score = max([r.get('score', 0) for r in results])
                    avg_score = sum([r.get('score', 0) for r in results]) / len(results)
                    vector_score = results[0].get('dense_score', 0) if results else 0
                    bm25_score = results[0].get('bm25_score', 0) if results else 0
                    
                    qlog.log_search_results(
                        num_candidates=num_candidates,
                        num_filtered=num_filtered,
                        top_score=top_score,
                        avg_score=avg_score,
                        vector_score=vector_score,
                        bm25_score=bm25_score,
                        duration_ms=search_duration
                    )
                
                if not results:
                    # Log no results
                    qlog.log_stage('no_results', duration_ms=search_duration, details={
                        'min_score': min_score,
                        'candidates': num_candidates
                    })
                    
                    return {
                        'answer': "Ich konnte keine relevanten Dokumente finden.",
                        'sources': [],
                        'context': [],
                        'method': 'no-results'
                    }
                
                # 2. Build context from retrieved chunks
                context_parts = []
                sources = []
                
                for result in results:
                    metadata = result['metadata']
                    sources.append({
                        'document_id': metadata['document_id'],
                        'title': metadata['title'],
                        'chunk': f"{metadata['chunk_index'] + 1}/{metadata['total_chunks']}",
                        'url': metadata.get('url', ''),
                        'score': round(result.get('score', 0), 4),
                        'dense_score': round(result.get('dense_score', 0), 4),
                        'bm25_score': round(result.get('bm25_score', 0), 4)
                    })
                    context_parts.append(
                        f"[{metadata['title']}]\n{result['text']}"
                    )
                
                context = "\n\n---\n\n".join(context_parts)
                
                # 3. Select prompt based on language
                language = self._detect_language(question)
                template = self.prompt_template_de if language == 'de' else self.prompt_template
                
                prompt = template.format(context=context, question=question)
                
                # 4. Generate answer
                llm_start = time.time()
                
                try:
                    logger.debug(f"Prompt length: {len(prompt)} chars")
                    
                    # Estimate input tokens (rough: 1 token ≈ 4 chars for English/German)
                    tokens_input = len(prompt) // 4
                    
                    answer = self.llm.invoke(prompt)
                    
                    llm_duration = (time.time() - llm_start) * 1000
                    
                    # Ensure proper encoding
                    if isinstance(answer, str):
                        answer = answer.encode('utf-8', errors='ignore').decode('utf-8')
                    
                    # Estimate output tokens
                    tokens_output = len(answer) // 4
                    tokens_per_sec = tokens_output / (llm_duration / 1000) if llm_duration > 0 else 0
                    
                    # Log LLM generation
                    qlog.log_llm_generation(
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        duration_ms=llm_duration,
                        tokens_per_sec=tokens_per_sec,
                        model=settings.OLLAMA_MODEL
                    )
                    
                    method = f'fast-rag-{settings.OLLAMA_MODEL}'
                    if metadata_filter:
                        method += '+filtered'
                        
                except Exception as e:
                    logger.error(f"LLM error: {e}")
                    answer = f"Fehler bei der Antwortgenerierung: {str(e)}"
                    method = 'error'
                    llm_duration = (time.time() - llm_start) * 1000
                    tokens_input = 0
                    tokens_output = 0
                    tokens_per_sec = 0
                
                # Calculate total duration
                total_duration = (time.time() - query_start_time) * 1000
                
                # Create metrics object
                query_metrics = QueryMetrics(
                    query_id=qlog.query_id,
                    query_text=question,
                    query_length=len(question),
                    total_duration_ms=total_duration,
                    rewrite_duration_ms=0,  # Not using rewrite yet
                    search_duration_ms=search_duration,
                    rerank_duration_ms=0,  # Not using reranking in this version
                    llm_duration_ms=llm_duration,
                    num_candidates=num_candidates,
                    num_filtered=num_filtered,
                    num_final=len(sources),
                    top_score=sources[0]['score'] if sources else 0,
                    avg_score=sum([s['score'] for s in sources]) / len(sources) if sources else 0,
                    min_score_threshold=min_score,
                    vector_score=sources[0]['dense_score'] if sources else 0,
                    bm25_score=sources[0]['bm25_score'] if sources else 0,
                    hybrid_dense_weight=settings.HYBRID_DENSE_WEIGHT,
                    hybrid_sparse_weight=settings.HYBRID_SPARSE_WEIGHT,
                    rerank_enabled=False,
                    model_name=settings.OLLAMA_MODEL,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    tokens_per_sec=tokens_per_sec,
                    cache_hit=False,  # This is a fresh query
                    has_answer=(method != 'error' and method != 'no-results'),
                    answer_length=len(answer),
                    num_sources=len(sources)
                )
                
                # Store metrics
                metrics_collector.record_query(query_metrics)
                metrics_storage.store_query_metrics(query_metrics)
                
                # Prepare result
                result = {
                    'answer': answer,
                    'sources': sources,
                    'context': context_parts,
                    'method': method,
                    'metrics': {
                        'total_duration_ms': round(total_duration, 2),
                        'search_duration_ms': round(search_duration, 2),
                        'llm_duration_ms': round(llm_duration, 2),
                        'tokens_per_sec': round(tokens_per_sec, 1),
                        'num_sources': len(sources),
                        'top_score': round(sources[0]['score'], 3) if sources else 0,
                        'cache_hit': False
                    }
                }
                
                # Cache the result
                if cache and method not in ['error', 'no-results']:
                    cache.set(question, result)
                    logger.debug(f"Cached query result: '{question}'")
                
                return result
                
            except Exception as e:
                # Log error
                logger.error(f"Query error: {e}", exc_info=True)
                return {
                    'answer': f"Ein Fehler ist aufgetreten: {str(e)}",
                    'sources': [],
                    'context': [],
                    'method': 'error'
                }
    
    def stream_query(self, question: str, n_results: int = None, metadata_filter: Dict = None) -> Generator[Dict, None, None]:
        """
        Stream answer tokens for perceived speed with comprehensive monitoring.
        
        Yields:
            Dicts with 'type' and 'data' fields:
            - {'type': 'token', 'data': 'text chunk'}
            - {'type': 'metadata', 'data': {...}}  # Final metadata
            
        Args:
            question: User query
            n_results: Number of results to retrieve (default: settings.TOP_K)
            metadata_filter: Optional Qdrant filter (e.g., {"document_id": 123})
        """
        if n_results is None:
            n_results = settings.TOP_K
        
        # Start monitoring
        query_start = time.time()
        
        with QueryLogger(question) as qlog:
            try:
                # SEARCH PHASE
                search_start = time.time()
                
                # Apply synonym expansion if enabled
                expanded_question = question
                if settings.ENABLE_SYNONYM_EXPANSION and should_expand_query(question):
                    expanded_question = expand_query_with_synonyms(question)
                    logger.info(f"Synonym expansion (streaming): '{question}' → '{expanded_question}'")
                
                # Use hybrid or dense search
                if self.use_hybrid and self.retriever:
                    results = self.retriever.search(
                        query=expanded_question,
                        n_results=n_results,
                        metadata_filter=metadata_filter or {}
                    )
                    method = "hybrid"
                else:
                    results = self.vector_store.search(
                        expanded_question, 
                        n_results=n_results,
                        metadata_filter=metadata_filter
                    )
                    method = "dense"
                
                search_duration = (time.time() - search_start) * 1000
                
                # Log search results
                if results:
                    qlog.log_search_results(
                        num_candidates=len(results),
                        num_filtered=len(results),
                        top_score=results[0].get('score', 0),
                        avg_score=sum(r.get('score', 0) for r in results) / len(results),
                        duration_ms=search_duration
                    )
                
                if not results:
                    answer = "Ich konnte keine relevanten Dokumente finden."
                    yield {'type': 'token', 'data': answer}
                    yield {
                        'type': 'metadata',
                        'data': {
                            'answer': answer,
                            'sources': [],
                            'method': 'no-results',
                            'metrics': {
                                'search_duration_ms': round(search_duration, 2),
                                'total_duration_ms': round((time.time() - query_start) * 1000, 2)
                            }
                        }
                    }
                    return
                
                # Build context
                context_parts = []
                sources = []
                
                for result in results:
                    metadata = result.get('metadata', {})
                    sources.append({
                        'document_id': metadata.get('document_id', 0),
                        'title': metadata.get('title', 'Unknown'),
                        'url': metadata.get('url', ''),
                        'score': round(result.get('score', 0), 4),
                        'vector_score': round(result.get('vector_score', 0), 4) if 'vector_score' in result else None,
                        'bm25_score': round(result.get('bm25_score', 0), 4) if 'bm25_score' in result else None
                    })
                    context_parts.append(f"[{metadata.get('title', 'Unknown')}]\n{result['text']}")
                
                context = "\n\n---\n\n".join(context_parts)
                
                # LLM GENERATION PHASE
                language = self._detect_language(question)
                template = self.prompt_template_de if language == 'de' else self.prompt_template
                prompt = template.format(context=context, question=question)
                
                llm_start = time.time()
                full_answer = ""
                token_count = 0
                
                try:
                    # Stream tokens
                    for chunk in self.llm.stream(prompt):
                        full_answer += chunk
                        token_count += 1
                        yield {'type': 'token', 'data': chunk}
                    
                    llm_duration = (time.time() - llm_start) * 1000
                    tokens_per_sec = token_count / (llm_duration / 1000) if llm_duration > 0 else 0
                    
                    # Log LLM generation
                    qlog.log_llm_generation(
                        tokens_input=len(prompt.split()),  # Rough estimate
                        tokens_output=token_count,
                        duration_ms=llm_duration,
                        tokens_per_sec=tokens_per_sec,
                        model=settings.OLLAMA_MODEL
                    )
                    
                    # Calculate total duration
                    total_duration = (time.time() - query_start) * 1000
                    
                    # Collect metrics
                    query_metrics = QueryMetrics(
                        query_id=qlog.query_id,
                        query_text=question,
                        query_length=len(question),
                        total_duration_ms=total_duration,
                        search_duration_ms=search_duration,
                        llm_duration_ms=llm_duration,
                        num_candidates=len(results),
                        num_filtered=len(results),
                        num_final=len(sources),
                        num_sources=len(sources),
                        top_score=sources[0]['score'] if sources else 0,
                        avg_score=sum(s['score'] for s in sources) / len(sources) if sources else 0,
                        model_name=settings.OLLAMA_MODEL,
                        tokens_input=len(prompt.split()),  # Rough estimate
                        tokens_output=token_count,
                        tokens_per_sec=tokens_per_sec,
                        has_answer=len(full_answer) > 0,
                        answer_length=len(full_answer)
                    )
                    
                    metrics_collector.record_query(query_metrics)
                    metrics_storage.store_query_metrics(query_metrics)
                    
                    # Yield final metadata
                    yield {
                        'type': 'metadata',
                        'data': {
                            'answer': full_answer,
                            'sources': sources,
                            'context': context_parts[:3],
                            'method': f'{method}-{settings.OLLAMA_MODEL}',
                            'metrics': {
                                'search_duration_ms': round(search_duration, 2),
                                'llm_duration_ms': round(llm_duration, 2),
                                'total_duration_ms': round(total_duration, 2),
                                'tokens_per_sec': round(tokens_per_sec, 1),
                                'num_sources': len(sources),
                                'top_score': round(sources[0]['score'], 3) if sources else 0
                            }
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_msg = f"\n\nFehler: {str(e)}"
                    yield {'type': 'token', 'data': error_msg}
                    yield {
                        'type': 'metadata',
                        'data': {
                            'answer': full_answer + error_msg,
                            'sources': sources,
                            'method': 'error'
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Query error: {e}", exc_info=True)
                error_msg = f"Ein Fehler ist aufgetreten: {str(e)}"
                yield {'type': 'token', 'data': error_msg}
                yield {
                    'type': 'metadata',
                    'data': {
                        'answer': error_msg,
                        'sources': [],
                        'method': 'error'
                    }
                }

