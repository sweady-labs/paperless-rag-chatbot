from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from typing import List, Dict
import os
from dotenv import load_dotenv
from .reranker import Reranker

load_dotenv()

class QueryEngine:
    def __init__(self, vector_store, use_reranker: bool = True):
        """Optimized query engine with better German support and performance."""
        self.vector_store = vector_store
        self.use_reranker = use_reranker
        
        # Check if using BGE-M3 hybrid
        self.is_hybrid = hasattr(vector_store, 'model') and hasattr(vector_store.model, 'encode')
        
        # Initialize Ollama with better model for German
        model = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
        self.llm = Ollama(
            model=model,
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            temperature=0.1,
            num_predict=512,  # Limit response length for faster generation
            timeout=90,  # 90 second timeout for LLM calls
            # Enforce German/English responses
            system="You are a helpful assistant. Always respond in the same language as the question. Never use Chinese characters."
        )
        
        # Initialize reranker
        if self.use_reranker:
            self.reranker = Reranker()
            print(f"✅ Query Engine initialized with {model} + Reranker")
        else:
            print(f"✅ Query Engine initialized with {model}")
        
        if self.is_hybrid:
            print("✅ Using BGE-M3 hybrid search (dense + sparse + colbert)")
        
        # Optimized RAG prompt template - clearer instructions
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Du bist ein hilfreicher Assistent, der Fragen basierend auf Dokumenten aus einem Paperless-NGX System beantwortet.

Nutze den folgenden Kontext aus relevanten Dokumenten um die Frage zu beantworten. Wenn die Antwort nicht im Kontext zu finden ist, sage das klar und deutlich.

WICHTIG: Antworte in der gleichen Sprache wie die Frage gestellt wurde. Nutze NIE chinesische Zeichen.

Kontext aus Dokumenten:
{context}

Frage: {question}

Antwort:"""
        )
    
    def query(self, question: str, n_results: int = 5, fast_rerank: bool = True) -> Dict:
        """
        Answer a question using optimized RAG pipeline.
        
        Args:
            question: User question
            n_results: Number of final results (default: 5 for faster responses)
            fast_rerank: Use fast reranking (recommended)
            
        Returns:
            Dict with 'answer', 'sources', 'context', 'method'
        """
        # 1. Retrieve with BGE-M3 hybrid search (reduced multiplier for speed)
        initial_k = min(n_results * 2, 15) if self.use_reranker else n_results
        
        if self.is_hybrid:
            results = self.vector_store.search(question, n_results=initial_k)
            method = 'bge-m3-hybrid'
        else:
            results = self.vector_store.search(question, n_results=initial_k)
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
            answer = self.llm.invoke(prompt)
            # Ensure proper encoding
            if isinstance(answer, str):
                answer = answer.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            answer = "Es gab einen Fehler bei der Generierung der Antwort."
        
        return {
            'answer': answer,
            'sources': sources,
            'context': context_parts,
            'method': method
        }
