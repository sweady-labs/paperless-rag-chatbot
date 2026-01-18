"""
Test script for monitoring infrastructure.

Tests:
- Structured logging
- Metrics collection
- Metrics storage
"""

import time
from src.monitoring.logger import QueryLogger, get_logger
from src.monitoring.metrics import QueryMetrics, get_metrics_collector
from src.monitoring.storage import MetricsStorage

def test_logging():
    """Test structured logging."""
    print("Testing logging...")
    
    with QueryLogger("Was ist die Hauptstadt von Deutschland?") as qlog:
        # Log query rewriting
        qlog.log_rewrite(
            original="Was ist die Hauptstadt von Deutschland?",
            rewritten="Hauptstadt Deutschland",
            patterns_applied=["^was\\s+", "\\s+von\\s+"],
            duration_ms=1.5
        )
        
        # Log search results
        qlog.log_search_results(
            num_candidates=10,
            num_filtered=5,
            top_score=0.85,
            avg_score=0.72,
            vector_score=0.80,
            bm25_score=0.90,
            duration_ms=145.3
        )
        
        # Log LLM generation
        qlog.log_llm_generation(
            tokens_input=512,
            tokens_output=89,
            duration_ms=2847.5,
            tokens_per_sec=31.2,
            model="llama3.1:8b"
        )
    
    print("✅ Logging test complete - check ./logs/query_processing.log")


def test_metrics():
    """Test metrics collection."""
    print("\nTesting metrics collection...")
    
    collector = get_metrics_collector()
    
    # Simulate 10 queries
    for i in range(10):
        metrics = QueryMetrics(
            query_id=f"test-{i}",
            query_text=f"Test query {i}",
            query_length=len(f"Test query {i}"),
            total_duration_ms=1000 + i * 100,
            rewrite_duration_ms=2.0,
            search_duration_ms=150.0,
            llm_duration_ms=800.0 + i * 100,
            num_candidates=10,
            num_filtered=5,
            num_final=3,
            top_score=0.85,
            avg_score=0.72,
            vector_score=0.80,
            bm25_score=0.90,
            model_name="llama3.1:8b",
            tokens_input=512,
            tokens_output=89,
            tokens_per_sec=30.0,
            cache_hit=(i % 3 == 0),  # 33% cache hit rate
            has_answer=True,
            answer_length=200,
            num_sources=3
        )
        collector.record_query(metrics)
    
    # Get statistics
    stats = collector.get_statistics()
    
    print(f"\n📊 Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Avg latency: {stats['latency']['mean']:.1f}ms")
    print(f"  P95 latency: {stats['latency']['p95']:.1f}ms")
    print(f"  Cache hit rate: {stats['cache']['hit_rate']}%")
    print(f"  Success rate: {stats['quality']['success_rate']}%")
    print(f"  Avg top score: {stats['quality']['avg_top_score']}")
    
    print("✅ Metrics collection test complete")


def test_storage():
    """Test metrics storage."""
    print("\nTesting metrics storage...")
    
    storage = MetricsStorage('./data/test_metrics.db')
    
    # Store some metrics
    for i in range(5):
        metrics = QueryMetrics(
            query_id=f"db-test-{i}",
            query_text=f"Database test query {i}",
            query_length=len(f"Database test query {i}"),
            total_duration_ms=1500 + i * 200,
            search_duration_ms=200.0,
            llm_duration_ms=1200.0 + i * 200,
            top_score=0.90 - i * 0.05,
            model_name="llama3.1:8b",
            tokens_input=600,
            tokens_output=95,
            tokens_per_sec=28.5,
            has_answer=True,
            num_sources=3
        )
        storage.store_query_metrics(metrics)
    
    # Get statistics from database
    stats = storage.get_statistics()
    
    print(f"\n📊 Database Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Avg latency: {stats['latency']['avg_ms']:.1f}ms")
    print(f"  Success rate: {stats['quality']['success_rate']}%")
    
    # Get recent queries
    recent = storage.get_recent_queries(limit=3)
    print(f"\n📜 Recent queries:")
    for q in recent:
        print(f"  - {q['query_text']}: {q['total_duration_ms']:.1f}ms")
    
    print("✅ Storage test complete - check ./data/test_metrics.db")


if __name__ == "__main__":
    print("="*60)
    print("MONITORING INFRASTRUCTURE TEST")
    print("="*60)
    
    test_logging()
    test_metrics()
    test_storage()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\nFiles created:")
    print("  - ./logs/query_processing.log (structured JSON logs)")
    print("  - ./data/test_metrics.db (SQLite metrics database)")
