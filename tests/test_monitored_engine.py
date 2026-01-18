"""
Test the monitored query engine with real queries.
"""

from src.rag.fast_vector_store import FastVectorStore
from src.rag.fast_query_engine import FastQueryEngine
from src.monitoring.metrics import get_metrics_collector
from src.monitoring.storage import MetricsStorage
import time

def test_monitored_queries():
    """Test query engine with monitoring."""
    print("="*60)
    print("TESTING MONITORED QUERY ENGINE")
    print("="*60)
    
    # Initialize
    print("\nInitializing vector store and query engine...")
    vector_store = FastVectorStore()
    engine = FastQueryEngine(vector_store)
    
    # Test queries
    test_queries = [
        "Wann ist Mavi geboren?",
        "Lohnsteuerbescheinigung 2024",
        "Welche Rechnungen gibt es von SWE?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Testing: {query}")
        print("-" * 60)
        
        start = time.time()
        result = engine.query(query, min_score=0.40)
        duration = time.time() - start
        
        print(f"✅ Query completed in {duration:.2f}s")
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"  📊 Metrics:")
            print(f"     - Total: {metrics['total_duration_ms']:.1f}ms")
            print(f"     - Search: {metrics['search_duration_ms']:.1f}ms")
            print(f"     - LLM: {metrics['llm_duration_ms']:.1f}ms")
            print(f"     - Speed: {metrics['tokens_per_sec']:.1f} tok/s")
            print(f"     - Sources: {metrics['num_sources']}")
            print(f"     - Top Score: {metrics['top_score']:.3f}")
        
        # Show answer preview
        answer = result.get('answer', 'No answer')[:150]
        print(f"  💬 Answer: {answer}...")
        
        results.append(result)
    
    # Get aggregated statistics
    print("\n" + "="*60)
    print("AGGREGATED STATISTICS")
    print("="*60)
    
    collector = get_metrics_collector()
    stats = collector.get_statistics()
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Avg latency: {stats['latency']['mean']:.1f}ms")
    print(f"  P50: {stats['latency']['p50']:.1f}ms")
    print(f"  P95: {stats['latency']['p95']:.1f}ms")
    print(f"  P99: {stats['latency']['p99']:.1f}ms")
    
    print(f"\n🎯 Quality Metrics:")
    print(f"  Success rate: {stats['quality']['success_rate']}%")
    print(f"  Avg top score: {stats['quality']['avg_top_score']}")
    print(f"  Avg sources: {stats['quality']['avg_num_sources']}")
    
    print(f"\n🤖 LLM Performance:")
    print(f"  Total input tokens: {stats['tokens']['total_input']}")
    print(f"  Total output tokens: {stats['tokens']['total_output']}")
    print(f"  Avg speed: {stats['tokens']['avg_per_sec']:.1f} tok/s")
    
    # Check database storage
    print("\n" + "="*60)
    print("DATABASE STORAGE")
    print("="*60)
    
    storage = MetricsStorage()
    db_stats = storage.get_statistics()
    
    print(f"\n📦 Stored Metrics:")
    print(f"  Total queries in DB: {db_stats['total_queries']}")
    print(f"  Avg latency: {db_stats['latency']['avg_ms']:.1f}ms")
    
    # Show recent queries
    recent = storage.get_recent_queries(limit=3)
    print(f"\n📜 Recent Queries (from DB):")
    for q in recent:
        print(f"  - \"{q['query_text']}\": {q['total_duration_ms']:.1f}ms (score: {q['top_score']:.3f})")
    
    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)
    print("\nCheck logs:")
    print("  - ./logs/query_processing.log (detailed JSON logs)")
    print("  - ./data/metrics.db (SQLite database)")


if __name__ == "__main__":
    test_monitored_queries()
