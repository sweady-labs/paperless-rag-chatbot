from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Optional
from src.config import settings
from .reranker import Reranker

class QueryEngine:
    def __init__(self, vector_store, use_reranker: bool = False):
        """Optimized query engine with better German support and performance."""
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        
        # Check if using BGE-M3 hybrid
        self.is_hybrid = hasattr(vector_store, 'model') and hasattr(vector_store.model, 'encode')
        
        # Initialize Ollama with configured model
        model = settings.OLLAMA_MODEL
        self.llm = Ollama(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.0,
            num_predict=256,  # Increased back to 256 for complete answers
            timeout=60,  # 60s is enough with faster llama3.2:3b model
            top_k=10,
            top_p=0.9,
            # Stronger system prompt
            system="You are a document assistant. You MUST answer ONLY based on the provided context. Never make up information. If the context doesn't contain the answer, say so clearly."
        )
        
        # Initialize reranker
        if self.use_reranker:
            self.reranker = Reranker()
            logger = __import__('logging').getLogger(__name__)
            logger.info(f"Query Engine initialized with {model} + Reranker")
        else:
            logger = __import__('logging').getLogger(__name__)
            logger.info(f"Query Engine initialized with {model}")
        
        if self.is_hybrid:
            logger.info("Using BGE-M3 hybrid search (dense + sparse + colbert)")
        
        # Optimized RAG prompt template - clearer instructions
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Du bist ein Assistent, der Fragen NUR basierend auf den bereitgestellten Dokumenten beantwortet.

=== WICHTIGE REGELN ===
1. Beantworte die Frage NUR mit Informationen aus dem unten stehenden Kontext
2. Wenn die Antwort NICHT im Kontext steht, sage: "Diese Information ist in den Dokumenten nicht vorhanden."
3. Erfinde NIEMALS Informationen - nutze NUR was im Kontext steht
4. Antworte in der gleichen Sprache wie die Frage
5. Zitiere direkt aus den Dokumenten wenn möglich

=== KONTEXT AUS DOKUMENTEN ===
{context}

=== FRAGE ===
{question}

=== ANTWORT (NUR BASIEREND AUF DEM KONTEXT OBEN) ===
"""
        )
    
    def query(self, question: str, n_results: int = 1, fast_rerank: bool = True) -> Dict:
        """
        Answer a question using optimized RAG pipeline.
        
        Args:
            question: User question
            n_results: Number of final results (default: 1 for maximum speed)
            fast_rerank: Use fast reranking (recommended)
            
        Returns:
            Dict with 'answer', 'sources', 'context', 'method'
        """
        # Detect if this is an analytical query requiring many documents
        analytical_keywords = ['alle', 'gesamt', 'pro monat', 'übersicht', 'tabelle', 'liste', 'jeden monat',
                              'all', 'total', 'overview', 'table', 'list', 'per month', 'each month', 
                              'summiere', 'sum', 'zusammen']
        is_analytical = any(keyword in question.lower() for keyword in analytical_keywords)
        
        # Extract metadata filters from query (simple keyword detection)
        metadata_filter = self._extract_metadata_filter(question)
        
        # 1. Retrieve with BGE-M3 hybrid search
        if is_analytical:
            # For analytical queries: retrieve MORE documents (up to 40)
            initial_k = min(n_results * 4, 40) if self.use_reranker else min(n_results * 3, 30)
        else:
            # Standard queries: keep it minimal for speed
            initial_k = min(n_results * 2, 10) if self.use_reranker else n_results
        
        if self.is_hybrid:
            results = self.vector_store.search(question, n_results=initial_k, metadata_filter=metadata_filter)
            method = 'bge-m3-hybrid'
            if metadata_filter:
                method += '+filtered'
        else:
            results = self.vector_store.search(question, n_results=initial_k, metadata_filter=metadata_filter)
            method = 'dense-only'
        
        if not results:
            return {
                'answer': "Ich konnte keine relevanten Dokumente finden, um deine Frage zu beantworten.",
                'sources': [],
                'context': [],
                'method': 'none'
            }
        
        # 2. Reranking
        if self.use_reranker:
            if fast_rerank:
                results = self.reranker.fast_rerank(question, results, top_k=n_results)
                method += '+fast_rerank'
            else:
                results = self.reranker.rerank(question, results, top_k=n_results)
                method += '+llm_rerank'
        
        # 3. Build context
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            score_info = {
                'document_id': result['metadata']['document_id'],
                'title': result['metadata']['title'],
                'chunk': f"{result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}",
                'url': result['metadata'].get('url', ''),  # Include URL for clickable links
            }
            
            # Add scores
            if 'hybrid_score' in result:
                score_info['score'] = round(result['hybrid_score'], 4)
                score_info['dense'] = round(result.get('dense_score', 0), 4)
                score_info['sparse'] = round(result.get('sparse_score', 0), 4)
            elif 'rerank_score' in result:
                score_info['score'] = round(result['rerank_score'], 4)
            else:
                score_info['score'] = round(result.get('score', 0), 4)
            
            sources.append(score_info)
            context_parts.append(
                f"[Dokument: {result['metadata']['title']}]\n{result['text']}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # 4. Generate answer with UTF-8 enforcement
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Calling LLM with model: {settings.OLLAMA_MODEL}")
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            answer = self.llm.invoke(prompt)
            
            # Ensure proper encoding
            if isinstance(answer, str):
                answer = answer.encode('utf-8', errors='ignore').decode('utf-8')
                
            logger.info(f"LLM response received: {len(answer)} chars")
        except Exception as e:
            import logging
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating answer: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            print(f"❌ Error generating answer: {e}")
            print(f"Full error:\n{traceback.format_exc()}")
            answer = f"Es gab einen Fehler bei der Generierung der Antwort: {str(e)}"
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context_parts,
            'method': method
        }
    
    def _extract_metadata_filter(self, question: str) -> Optional[Dict]:
        """Extract metadata filters from question for faster retrieval."""
        import re
        filter_dict = {}
        
        # Extract correspondent names (Amazon, REWE, etc.)
        # Common patterns: "bei Amazon", "von Amazon", "at Amazon", "from Amazon"
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
                # Only add if it looks like a company name (capitalized)
                if len(correspondent) > 2:
                    filter_dict['correspondent'] = correspondent
                    break
        
        # Note: Date filtering would require parsing dates from created/modified fields
        # This is complex and better handled by improving the query itself
        
        return filter_dict if filter_dict else None
