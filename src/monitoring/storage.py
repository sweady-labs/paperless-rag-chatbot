"""
SQLite-based storage for query metrics and search analytics.

Features:
- Time-series metrics storage
- Efficient querying with indexes
- Automatic schema creation
- Data retention policies
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager

from .metrics import QueryMetrics, SearchMetrics


class MetricsStorage:
    """
    SQLite storage for query and search metrics.
    
    Usage:
        storage = MetricsStorage('data/metrics.db')
        storage.store_query_metrics(metrics)
        stats = storage.get_statistics(last_hours=24)
    """
    
    def __init__(self, db_path: str = './data/metrics.db'):
        """
        Initialize metrics storage.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Query metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    query_text TEXT,
                    query_length INTEGER,
                    
                    -- Timing (milliseconds)
                    total_duration_ms REAL,
                    rewrite_duration_ms REAL,
                    search_duration_ms REAL,
                    rerank_duration_ms REAL,
                    llm_duration_ms REAL,
                    
                    -- Search metrics
                    num_candidates INTEGER,
                    num_filtered INTEGER,
                    num_final INTEGER,
                    top_score REAL,
                    avg_score REAL,
                    min_score_threshold REAL,
                    
                    -- Hybrid search
                    vector_score REAL,
                    bm25_score REAL,
                    hybrid_dense_weight REAL,
                    hybrid_sparse_weight REAL,
                    
                    -- Reranking
                    rerank_enabled BOOLEAN,
                    rerank_score_improvement REAL,
                    top1_changed BOOLEAN,
                    
                    -- LLM
                    model_name TEXT,
                    tokens_input INTEGER,
                    tokens_output INTEGER,
                    tokens_per_sec REAL,
                    
                    -- Cache
                    cache_hit BOOLEAN,
                    
                    -- Results
                    has_answer BOOLEAN,
                    answer_length INTEGER,
                    num_sources INTEGER
                )
            ''')
            
            # Search metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    
                    -- Vector search
                    vector_candidates INTEGER,
                    vector_duration_ms REAL,
                    vector_top_score REAL,
                    vector_avg_score REAL,
                    
                    -- BM25 search
                    bm25_candidates INTEGER,
                    bm25_duration_ms REAL,
                    bm25_top_score REAL,
                    bm25_avg_score REAL,
                    
                    -- Fusion
                    fusion_duration_ms REAL,
                    fusion_method TEXT,
                    
                    -- Filtering
                    pre_filter_count INTEGER,
                    post_filter_count INTEGER,
                    filter_threshold REAL
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_query_timestamp 
                ON query_metrics(timestamp DESC)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_query_id 
                ON query_metrics(query_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_search_query_id 
                ON search_metrics(query_id)
            ''')
    
    def store_query_metrics(self, metrics: QueryMetrics):
        """
        Store query metrics.
        
        Args:
            metrics: QueryMetrics instance
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO query_metrics (
                    query_id, timestamp, query_text, query_length,
                    total_duration_ms, rewrite_duration_ms, search_duration_ms,
                    rerank_duration_ms, llm_duration_ms,
                    num_candidates, num_filtered, num_final,
                    top_score, avg_score, min_score_threshold,
                    vector_score, bm25_score, hybrid_dense_weight, hybrid_sparse_weight,
                    rerank_enabled, rerank_score_improvement, top1_changed,
                    model_name, tokens_input, tokens_output, tokens_per_sec,
                    cache_hit, has_answer, answer_length, num_sources
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?
                )
            ''', (
                metrics.query_id,
                metrics.timestamp.isoformat(),
                metrics.query_text,
                metrics.query_length,
                metrics.total_duration_ms,
                metrics.rewrite_duration_ms,
                metrics.search_duration_ms,
                metrics.rerank_duration_ms,
                metrics.llm_duration_ms,
                metrics.num_candidates,
                metrics.num_filtered,
                metrics.num_final,
                metrics.top_score,
                metrics.avg_score,
                metrics.min_score_threshold,
                metrics.vector_score,
                metrics.bm25_score,
                metrics.hybrid_dense_weight,
                metrics.hybrid_sparse_weight,
                metrics.rerank_enabled,
                metrics.rerank_score_improvement,
                metrics.top1_changed,
                metrics.model_name,
                metrics.tokens_input,
                metrics.tokens_output,
                metrics.tokens_per_sec,
                metrics.cache_hit,
                metrics.has_answer,
                metrics.answer_length,
                metrics.num_sources,
            ))
    
    def store_search_metrics(self, metrics: SearchMetrics):
        """
        Store search metrics.
        
        Args:
            metrics: SearchMetrics instance
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO search_metrics (
                    query_id, timestamp,
                    vector_candidates, vector_duration_ms, vector_top_score, vector_avg_score,
                    bm25_candidates, bm25_duration_ms, bm25_top_score, bm25_avg_score,
                    fusion_duration_ms, fusion_method,
                    pre_filter_count, post_filter_count, filter_threshold
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.query_id,
                metrics.timestamp.isoformat(),
                metrics.vector_candidates,
                metrics.vector_duration_ms,
                metrics.vector_top_score,
                metrics.vector_avg_score,
                metrics.bm25_candidates,
                metrics.bm25_duration_ms,
                metrics.bm25_top_score,
                metrics.bm25_avg_score,
                metrics.fusion_duration_ms,
                metrics.fusion_method,
                metrics.pre_filter_count,
                metrics.post_filter_count,
                metrics.filter_threshold,
            ))
    
    def get_statistics(
        self,
        last_hours: Optional[int] = None,
        last_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics.
        
        Args:
            last_hours: Only consider queries from last N hours
            last_days: Only consider queries from last N days
        
        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build time filter
            time_filter = ""
            if last_hours:
                cutoff = datetime.utcnow() - timedelta(hours=last_hours)
                time_filter = f"WHERE timestamp >= '{cutoff.isoformat()}'"
            elif last_days:
                cutoff = datetime.utcnow() - timedelta(days=last_days)
                time_filter = f"WHERE timestamp >= '{cutoff.isoformat()}'"
            
            # Query count
            cursor.execute(f"SELECT COUNT(*) as count FROM query_metrics {time_filter}")
            total_queries = cursor.fetchone()['count']
            
            if total_queries == 0:
                return {'total_queries': 0, 'message': 'No queries in selected time range'}
            
            # Latency statistics
            cursor.execute(f'''
                SELECT 
                    AVG(total_duration_ms) as avg_latency,
                    MIN(total_duration_ms) as min_latency,
                    MAX(total_duration_ms) as max_latency
                FROM query_metrics {time_filter}
            ''')
            latency = cursor.fetchone()
            
            # Cache statistics
            cursor.execute(f'''
                SELECT 
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as hits,
                    SUM(CASE WHEN cache_hit = 0 THEN 1 ELSE 0 END) as misses
                FROM query_metrics {time_filter}
            ''')
            cache = cursor.fetchone()
            cache_hit_rate = (cache['hits'] / total_queries * 100) if total_queries > 0 else 0.0
            
            # Success rate
            cursor.execute(f'''
                SELECT 
                    SUM(CASE WHEN has_answer = 1 THEN 1 ELSE 0 END) as successful
                FROM query_metrics {time_filter}
            ''')
            success = cursor.fetchone()
            success_rate = (success['successful'] / total_queries * 100) if total_queries > 0 else 0.0
            
            # Average scores
            cursor.execute(f'''
                SELECT 
                    AVG(top_score) as avg_top_score,
                    AVG(num_sources) as avg_sources
                FROM query_metrics {time_filter}
                WHERE top_score > 0
            ''')
            scores = cursor.fetchone()
            
            # Token statistics
            cursor.execute(f'''
                SELECT 
                    SUM(tokens_input) as total_input,
                    SUM(tokens_output) as total_output,
                    AVG(tokens_per_sec) as avg_per_sec
                FROM query_metrics {time_filter}
            ''')
            tokens = cursor.fetchone()
            
            return {
                'total_queries': total_queries,
                'latency': {
                    'avg_ms': round(latency['avg_latency'], 2),
                    'min_ms': round(latency['min_latency'], 2),
                    'max_ms': round(latency['max_latency'], 2),
                },
                'cache': {
                    'hit_rate': round(cache_hit_rate, 1),
                    'hits': cache['hits'],
                    'misses': cache['misses'],
                },
                'quality': {
                    'success_rate': round(success_rate, 1),
                    'avg_top_score': round(scores['avg_top_score'] or 0, 3),
                    'avg_sources': round(scores['avg_sources'] or 0, 1),
                },
                'tokens': {
                    'total_input': tokens['total_input'],
                    'total_output': tokens['total_output'],
                    'avg_per_sec': round(tokens['avg_per_sec'] or 0, 1),
                },
            }
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent queries.
        
        Args:
            limit: Number of queries to return
        
        Returns:
            List of query dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM query_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_slow_queries(
        self,
        threshold_ms: float = 5000,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get slowest queries above threshold.
        
        Args:
            threshold_ms: Minimum duration
            limit: Number of queries to return
        
        Returns:
            List of slow query dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM query_metrics
                WHERE total_duration_ms > ?
                ORDER BY total_duration_ms DESC
                LIMIT ?
            ''', (threshold_ms, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days: int = 90):
        """
        Remove metrics older than specified days.
        
        Args:
            days: Keep data newer than this many days
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Delete old query metrics
            cursor.execute('''
                DELETE FROM query_metrics
                WHERE timestamp < ?
            ''', (cutoff.isoformat(),))
            
            deleted_queries = cursor.rowcount
            
            # Delete old search metrics
            cursor.execute('''
                DELETE FROM search_metrics
                WHERE timestamp < ?
            ''', (cutoff.isoformat(),))
            
            deleted_searches = cursor.rowcount
            
            # Vacuum to reclaim space
            cursor.execute('VACUUM')
            
            return {
                'deleted_queries': deleted_queries,
                'deleted_searches': deleted_searches,
                'cutoff_date': cutoff.isoformat(),
            }
