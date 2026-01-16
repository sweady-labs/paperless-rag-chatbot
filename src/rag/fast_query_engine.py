"""
Fast Query Engine optimized for Apple Silicon.

Key optimizations:
- Uses qwen2.5:3b for fast inference (~35-50 tok/s on M5)
- Smaller context window (2048 tokens)
- Shorter max output (200 tokens)
- No reranking overhead
- Optional streaming for perceived speed

Author: Optimized for M5 GPU
"""

import logging
import re
from typing import List, Dict, Optional, Generator

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from src.config import settings

logger = logging.getLogger(__name__)


class FastQueryEngine:
    """
    Optimized query engine for fast RAG responses.
    
    Designed for sub-2-second end-to-end latency on Apple Silicon.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the fast query engine.
        
        Args:
            vector_store: FastVectorStore instance for retrieval
        """
        self.vector_store = vector_store
        
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
        
        # German prompt for German queries
        self.prompt_template_de = PromptTemplate(
            input_variables=["context", "question"],
            template="""Beantworte die Frage NUR basierend auf den Dokumenten unten. Wenn die Antwort nicht in den Dokumenten steht, sage das.

DOKUMENTE:
{context}

FRAGE: {question}

ANTWORT:"""
        )
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'von', 'mit', 'für', 'was', 'wie', 'wo', 'wann']
        words = text.lower().split()
        german_count = sum(1 for word in words if word in german_indicators)
        return 'de' if german_count >= 2 else 'en'
    
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
    
    def query(self, question: str, n_results: int = None) -> Dict:
        """
        Answer a question using fast RAG pipeline.
        
        Args:
            question: User question
            n_results: Number of chunks to retrieve (default from settings)
            
        Returns:
            Dict with 'answer', 'sources', 'context', 'method'
        """
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
        
        # 1. Retrieve relevant chunks
        results = self.vector_store.search(
            question, 
            n_results=n_results,
            metadata_filter=metadata_filter
        )
        
        if not results:
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
                'score': round(result.get('score', 0), 4)
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
        try:
            logger.debug(f"Prompt length: {len(prompt)} chars")
            answer = self.llm.invoke(prompt)
            
            # Ensure proper encoding
            if isinstance(answer, str):
                answer = answer.encode('utf-8', errors='ignore').decode('utf-8')
            
            method = f'fast-rag-{settings.OLLAMA_MODEL}'
            if metadata_filter:
                method += '+filtered'
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Fehler bei der Antwortgenerierung: {str(e)}"
            method = 'error'
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context_parts,
            'method': method
        }
    
    def stream_query(self, question: str, n_results: int = None) -> Generator[str, None, Dict]:
        """
        Stream answer tokens for perceived speed.
        
        Yields:
            Individual tokens as they're generated
            
        Returns:
            Final result dict (via StopIteration value)
        """
        if n_results is None:
            n_results = settings.TOP_K
        
        # Retrieve chunks
        results = self.vector_store.search(question, n_results=n_results)
        
        if not results:
            yield "Ich konnte keine relevanten Dokumente finden."
            return {
                'answer': "Ich konnte keine relevanten Dokumente finden.",
                'sources': [],
                'method': 'no-results'
            }
        
        # Build context
        context_parts = []
        sources = []
        
        for result in results:
            metadata = result['metadata']
            sources.append({
                'document_id': metadata['document_id'],
                'title': metadata['title'],
                'url': metadata.get('url', ''),
                'score': round(result.get('score', 0), 4)
            })
            context_parts.append(f"[{metadata['title']}]\n{result['text']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Select prompt
        language = self._detect_language(question)
        template = self.prompt_template_de if language == 'de' else self.prompt_template
        prompt = template.format(context=context, question=question)
        
        # Stream tokens
        full_answer = ""
        try:
            for chunk in self.llm.stream(prompt):
                full_answer += chunk
                yield chunk
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n\nFehler: {str(e)}"
            full_answer = f"Fehler: {str(e)}"
        
        return {
            'answer': full_answer,
            'sources': sources,
            'method': f'streamed-{settings.OLLAMA_MODEL}'
        }
