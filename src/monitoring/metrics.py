"""
Metrics collection for query performance and quality tracking.

Tracks:
- Query latency (p50, p95, p99)
- Stage-wise timing breakdown
- Search quality metrics
- System performance
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import statistics


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    
    # Identifiers
    query_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Query info
    query_text: str = ""
    query_length: int = 0
    
    # Timing breakdown (milliseconds)
    total_duration_ms: float = 0.0
    rewrite_duration_ms: float = 0.0
    search_duration_ms: float = 0.0
    rerank_duration_ms: float = 0.0
    llm_duration_ms: float = 0.0
    
    # Search metrics
    num_candidates: int = 0
    num_filtered: int = 0
    num_final: int = 0
    top_score: float = 0.0
    avg_score: float = 0.0
    min_score_threshold: float = 0.0
    
    # Hybrid search details
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_dense_weight: float = 0.6
    hybrid_sparse_weight: float = 0.4
    
    # Reranking
    rerank_enabled: bool = False
    rerank_score_improvement: float = 0.0
    top1_changed: bool = False
    
    # LLM metrics
    model_name: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_per_sec: float = 0.0
    
    # Cache
    cache_hit: bool = False
    
    # Result quality
    has_answer: bool = False
    answer_length: int = 0
    num_sources: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SearchMetrics:
    """Detailed metrics for a search operation."""
    
    query_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Vector search
    vector_candidates: int = 0
    vector_duration_ms: float = 0.0
    vector_top_score: float = 0.0
    vector_avg_score: float = 0.0
    
    # BM25 search
    bm25_candidates: int = 0
    bm25_duration_ms: float = 0.0
    bm25_top_score: float = 0.0
    bm25_avg_score: float = 0.0
    
    # Hybrid fusion
    fusion_duration_ms: float = 0.0
    fusion_method: str = "weighted"
    
    # Filtering
    pre_filter_count: int = 0
    post_filter_count: int = 0
    filter_threshold: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """
    Collects and aggregates metrics for monitoring and analysis.
    
    Usage:
        collector = MetricsCollector()
        
        # Record query metrics
        metrics = QueryMetrics(query_id="abc", total_duration_ms=1234)
        collector.record_query(metrics)
        
        # Get statistics
        stats = collector.get_statistics()
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of queries to keep in memory
        """
        self.max_history = max_history
        self.query_metrics: List[QueryMetrics] = []
        self.search_metrics: List[SearchMetrics] = []
        
        # Aggregated statistics
        self.total_queries = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0
        
    def record_query(self, metrics: QueryMetrics):
        """
        Record query metrics.
        
        Args:
            metrics: QueryMetrics instance
        """
        self.query_metrics.append(metrics)
        self.total_queries += 1
        
        if metrics.cache_hit:
            self.total_cache_hits += 1
        else:
            self.total_cache_misses += 1
        
        # Keep only recent history
        if len(self.query_metrics) > self.max_history:
            self.query_metrics.pop(0)
    
    def record_search(self, metrics: SearchMetrics):
        """
        Record search metrics.
        
        Args:
            metrics: SearchMetrics instance
        """
        self.search_metrics.append(metrics)
        
        # Keep only recent history
        if len(self.search_metrics) > self.max_history:
            self.search_metrics.pop(0)
    
    def get_statistics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregated statistics.
        
        Args:
            last_n: Only consider last N queries (default: all)
        
        Returns:
            Dictionary with statistics
        """
        if not self.query_metrics:
            return {
                'total_queries': 0,
                'message': 'No metrics collected yet'
            }
        
        # Select queries to analyze
        queries = self.query_metrics[-last_n:] if last_n else self.query_metrics
        
        # Extract durations
        total_durations = [q.total_duration_ms for q in queries]
        search_durations = [q.search_duration_ms for q in queries if q.search_duration_ms > 0]
        llm_durations = [q.llm_duration_ms for q in queries if q.llm_duration_ms > 0]
        
        # Calculate percentiles
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * p)
            return sorted_data[min(index, len(sorted_data) - 1)]
        
        # Cache statistics
        cache_hit_rate = (
            self.total_cache_hits / self.total_queries * 100
            if self.total_queries > 0 else 0.0
        )
        
        # Success rate
        successful = sum(1 for q in queries if q.has_answer)
        success_rate = successful / len(queries) * 100 if queries else 0.0
        
        # Average scores
        avg_top_score = statistics.mean([q.top_score for q in queries if q.top_score > 0]) if queries else 0.0
        avg_num_sources = statistics.mean([q.num_sources for q in queries]) if queries else 0.0
        
        # Token statistics
        total_input_tokens = sum(q.tokens_input for q in queries)
        total_output_tokens = sum(q.tokens_output for q in queries)
        avg_tokens_per_sec = statistics.mean([q.tokens_per_sec for q in queries if q.tokens_per_sec > 0]) if queries else 0.0
        
        return {
            'total_queries': self.total_queries,
            'analyzed_queries': len(queries),
            
            # Latency statistics (ms)
            'latency': {
                'mean': statistics.mean(total_durations) if total_durations else 0.0,
                'median': statistics.median(total_durations) if total_durations else 0.0,
                'p50': percentile(total_durations, 0.50),
                'p95': percentile(total_durations, 0.95),
                'p99': percentile(total_durations, 0.99),
                'min': min(total_durations) if total_durations else 0.0,
                'max': max(total_durations) if total_durations else 0.0,
            },
            
            # Stage breakdown
            'stages': {
                'search_avg_ms': statistics.mean(search_durations) if search_durations else 0.0,
                'llm_avg_ms': statistics.mean(llm_durations) if llm_durations else 0.0,
            },
            
            # Cache statistics
            'cache': {
                'hit_rate': round(cache_hit_rate, 1),
                'hits': self.total_cache_hits,
                'misses': self.total_cache_misses,
            },
            
            # Quality metrics
            'quality': {
                'success_rate': round(success_rate, 1),
                'avg_top_score': round(avg_top_score, 3),
                'avg_num_sources': round(avg_num_sources, 1),
            },
            
            # Token statistics
            'tokens': {
                'total_input': total_input_tokens,
                'total_output': total_output_tokens,
                'avg_per_sec': round(avg_tokens_per_sec, 1),
            },
        }
    
    def get_recent_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent query metrics.
        
        Args:
            n: Number of recent queries to return
        
        Returns:
            List of query metrics dictionaries
        """
        recent = self.query_metrics[-n:]
        return [q.to_dict() for q in recent]
    
    def get_slow_queries(self, threshold_ms: float = 5000, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get slowest queries above threshold.
        
        Args:
            threshold_ms: Minimum duration to consider slow
            n: Number of slow queries to return
        
        Returns:
            List of slow query metrics
        """
        slow = [
            q for q in self.query_metrics
            if q.total_duration_ms > threshold_ms
        ]
        slow.sort(key=lambda q: q.total_duration_ms, reverse=True)
        return [q.to_dict() for q in slow[:n]]
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        if not self.query_metrics:
            return {'message': 'No cache data yet'}
        
        recent = self.query_metrics[-100:]  # Last 100 queries
        recent_hits = sum(1 for q in recent if q.cache_hit)
        recent_hit_rate = recent_hits / len(recent) * 100 if recent else 0.0
        
        return {
            'total_hits': self.total_cache_hits,
            'total_misses': self.total_cache_misses,
            'overall_hit_rate': round(
                self.total_cache_hits / self.total_queries * 100
                if self.total_queries > 0 else 0.0,
                1
            ),
            'recent_hit_rate': round(recent_hit_rate, 1),
            'total_queries': self.total_queries,
        }
    
    def reset(self):
        """Reset all metrics."""
        self.query_metrics.clear()
        self.search_metrics.clear()
        self.total_queries = 0
        self.total_cache_hits = 0
        self.total_cache_misses = 0


# Global metrics collector instance
_global_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _global_collector
