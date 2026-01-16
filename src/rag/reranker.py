"""
Re-ranker for improving retrieval precision using cross-encoder models.
Uses BGE reranker models as recommended by BGE-M3 documentation.
"""
from typing import List, Dict
from src.config import settings

# Try to use BGE reranker for best performance
try:
    from FlagEmbedding import FlagReranker
    BGE_RERANKER_AVAILABLE = True
except ImportError:
    BGE_RERANKER_AVAILABLE = False
    # Fallback to LLM-based reranking
    from langchain_community.llms import Ollama


class Reranker:
    """
    Re-rank retrieved documents using cross-encoder models.
    
    As recommended by BGE-M3 authors:
    - Use bge-reranker-v2-m3 for best accuracy
    - Cross-encoders are more accurate than bi-encoders for ranking
    - Apply re-ranking after hybrid retrieval
    """
    
    def __init__(self, use_bge_reranker: bool = True):
        self.use_bge_reranker = use_bge_reranker and BGE_RERANKER_AVAILABLE
        
        if self.use_bge_reranker:
            # Use BGE reranker (recommended)
            reranker_model = settings.BGE_RERANKER_MODEL
            logger = __import__('logging').getLogger(__name__)
            logger.info(f"Loading BGE reranker: {reranker_model}")
            self.reranker = FlagReranker(
                reranker_model,
                use_fp16=True,  # Faster on Mac M-series
                device='cpu'
            )
            logger.info("BGE reranker loaded successfully!")
        else:
            # Fallback to LLM-based reranking
            logger = __import__('logging').getLogger(__name__)
            logger.warning("FlagEmbedding not available, using LLM-based reranking")
            self.llm = Ollama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=0.0  # Deterministic for ranking
            )
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank results using BGE reranker or LLM.
        
        Args:
            query: User query
            results: List of retrieved chunks with metadata
            top_k: Number of top results to return
            
        Returns:
            Re-ranked list of results
        """
        if not results:
            return []
        
        # If we have fewer results than top_k, just return them
        if len(results) <= top_k:
            return results
        
        if self.use_bge_reranker:
            # Use BGE reranker (fast and accurate)
            sentence_pairs = [[query, result['text']] for result in results]
            
            # Compute relevance scores
            scores = self.reranker.compute_score(
                sentence_pairs,
                max_length=8192  # BGE reranker supports long context
            )
            
            # Handle both single score and list of scores
            if isinstance(scores, (int, float)):
                scores = [scores]
            
            # Add reranker scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = score
                result['original_score'] = result.get('score', 0.5)
            
            # Sort by rerank score
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return results[:top_k]
        
        else:
            # Fallback: LLM-based re-ranking (slower)
            scored_results = []
            
            for result in results:
                # Create a relevance scoring prompt
                prompt = f"""Rate the relevance of this document chunk to the query on a scale of 0-10.
Only respond with a number.

Query: {query}

Document Chunk:
{result['text'][:500]}...

Relevance Score (0-10):"""
                
                try:
                    # Get relevance score from LLM
                    response = self.llm.invoke(prompt)
                    # Extract numeric score
                    score_str = response.strip().split()[0]
                    score = float(score_str)
                    
                    # Combine with original similarity score (weighted average)
                    # 70% LLM score, 30% embedding similarity
                    original_score = result.get('score', 0.5)
                    combined_score = 0.7 * (score / 10.0) + 0.3 * original_score
                    
                    result['rerank_score'] = combined_score
                    result['llm_relevance'] = score
                    scored_results.append(result)
                except Exception as e:
                    # If scoring fails, use original score
                    result['rerank_score'] = result.get('score', 0.5)
                    result['llm_relevance'] = 5.0  # neutral score
                    scored_results.append(result)
            
            # Sort by rerank score
            scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return scored_results[:top_k]
    
    def fast_rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Fast re-ranking using keyword overlap and position.
        Much faster than LLM-based reranking.
        
        Args:
            query: User query
            results: List of retrieved chunks
            top_k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        query_terms = set(query.lower().split())
        
        for result in results:
            text_lower = result['text'].lower()
            
            # Count keyword matches
            keyword_score = sum(1 for term in query_terms if term in text_lower)
            
            # Boost if keywords appear early in the text
            position_score = 0
            first_500 = text_lower[:500]
            if any(term in first_500 for term in query_terms):
                position_score = 0.2
            
            # Combine with original embedding score
            original_score = result.get('score', 0.5)
            combined_score = (
                0.5 * original_score +  # Embedding similarity
                0.3 * (keyword_score / len(query_terms)) +  # Keyword coverage
                0.2 * position_score  # Position bonus
            )
            
            result['rerank_score'] = combined_score
            result['keyword_matches'] = keyword_score
        
        # Sort by combined score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results[:top_k]
