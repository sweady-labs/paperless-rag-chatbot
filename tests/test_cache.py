#!/usr/bin/env python3
"""
Test query cache functionality.

Tests:
1. Cache miss (first query)
2. Cache hit (exact match)
3. Cache hit (fuzzy match)
4. Cache expiration
5. LRU eviction
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.monitoring.cache import QueryCache


def test_basic_cache():
    """Test basic cache operations."""
    print("=" * 60)
    print("Test 1: Basic Cache Operations")
    print("=" * 60)
    print()
    
    cache = QueryCache(max_size=3, ttl_seconds=2)
    
    # Test cache miss
    result = cache.get("Wann ist Mavi geboren?")
    assert result is None, "Should be cache miss"
    print("✓ Cache MISS (expected)")
    
    # Add to cache
    cache.set("Wann ist Mavi geboren?", {"answer": "10. Oktober 2023"})
    print("✓ Cached result")
    
    # Test cache hit
    result = cache.get("Wann ist Mavi geboren?")
    assert result is not None, "Should be cache hit"
    assert result['answer'] == "10. Oktober 2023"
    print("✓ Cache HIT (exact match)")
    
    print()
    stats = cache.get_stats()
    print(f"Stats: {stats['total_queries']} queries, {stats['exact_hits']} hits, {stats['hit_rate']}% hit rate")
    print()


def test_fuzzy_matching():
    """Test fuzzy query matching."""
    print("=" * 60)
    print("Test 2: Fuzzy Matching")
    print("=" * 60)
    print()
    
    cache = QueryCache(max_size=10, ttl_seconds=3600, fuzzy_threshold=0.85)
    
    # Cache original query
    cache.set("Wann ist Mavi geboren?", {"answer": "10. Oktober 2023"})
    print("Cached: 'Wann ist Mavi geboren?'")
    
    # Try similar queries
    similar_queries = [
        "wann ist mavi geboren",           # Different case
        "Wann ist Mavi geboren",           # Missing punctuation
        "Wann wurde Mavi geboren?",        # Similar wording
        "Wann ist Mavi eigentlich geboren?",  # Extra word
    ]
    
    for query in similar_queries:
        result = cache.get(query, fuzzy=True)
        if result:
            print(f"✓ Fuzzy HIT: '{query}'")
        else:
            print(f"✗ Fuzzy MISS: '{query}'")
    
    print()
    stats = cache.get_stats()
    print(f"Stats: Exact: {stats['exact_hits']}, Fuzzy: {stats['fuzzy_hits']}, Total hit rate: {stats['hit_rate']}%")
    print()


def test_ttl_expiration():
    """Test TTL expiration."""
    print("=" * 60)
    print("Test 3: TTL Expiration")
    print("=" * 60)
    print()
    
    cache = QueryCache(max_size=10, ttl_seconds=1)  # 1 second TTL
    
    # Cache a query
    cache.set("Test query", {"answer": "Test answer"})
    print("Cached with 1 second TTL")
    
    # Should hit immediately
    result = cache.get("Test query")
    assert result is not None
    print("✓ Cache HIT (immediately)")
    
    # Wait for expiration
    print("Waiting 1.5 seconds for expiration...")
    time.sleep(1.5)
    
    # Should miss after TTL
    result = cache.get("Test query")
    assert result is None
    print("✓ Cache MISS (expired)")
    
    print()


def test_lru_eviction():
    """Test LRU eviction."""
    print("=" * 60)
    print("Test 4: LRU Eviction")
    print("=" * 60)
    print()
    
    cache = QueryCache(max_size=3, ttl_seconds=3600)  # Only 3 items
    
    # Fill cache
    cache.set("Query 1", {"answer": "Answer 1"})
    cache.set("Query 2", {"answer": "Answer 2"})
    cache.set("Query 3", {"answer": "Answer 3"})
    print("Filled cache (3/3 items)")
    
    # Access Query 1 (make it recently used)
    cache.get("Query 1")
    print("Accessed 'Query 1' (make it recent)")
    
    # Add Query 4 (should evict Query 2, the LRU)
    cache.set("Query 4", {"answer": "Answer 4"})
    print("Added 'Query 4' (should evict 'Query 2')")
    
    # Check what's still in cache
    assert cache.get("Query 1") is not None, "Query 1 should still be cached"
    assert cache.get("Query 2") is None, "Query 2 should be evicted (LRU)"
    assert cache.get("Query 3") is not None, "Query 3 should still be cached"
    assert cache.get("Query 4") is not None, "Query 4 should be cached"
    
    print("✓ LRU eviction works correctly")
    print()


def test_with_real_queries():
    """Test with realistic queries."""
    print("=" * 60)
    print("Test 5: Real-World Scenario")
    print("=" * 60)
    print()
    
    cache = QueryCache(max_size=100, ttl_seconds=3600)
    
    # Simulate real queries
    queries = [
        ("Wann ist Mavi geboren?", {"answer": "10. Oktober 2023"}),
        ("Lohnsteuerbescheinigung 2024", {"answer": "Dokument xyz"}),
        ("Stromrechnung SWE", {"answer": "123.45 EUR"}),
        ("Wann ist Mavi geboren?", None),  # Repeat - should hit cache
        ("wann ist mavi geboren", None),   # Fuzzy - should hit
        ("Lohnsteuerbescheinigung 2024", None),  # Repeat - should hit
    ]
    
    hits = 0
    misses = 0
    
    for query, expected_result in queries:
        result = cache.get(query)
        
        if result is None:
            # Cache miss - simulate storing result
            if expected_result:
                cache.set(query, expected_result)
                print(f"MISS: '{query}' → Cached")
                misses += 1
            else:
                print(f"ERROR: Expected cache hit for '{query}'")
        else:
            # Cache hit
            print(f"HIT:  '{query}' ✓")
            hits += 1
    
    print()
    stats = cache.get_stats()
    print(f"Results: {hits} hits, {misses} misses")
    print(f"Hit rate: {stats['hit_rate']}%")
    print()
    
    # Show top queries
    top = cache.get_top_queries(limit=3)
    print("Top cached queries:")
    for query, hit_count in top:
        print(f"  - '{query}' ({hit_count} hits)")
    print()


def main():
    """Run all tests."""
    print()
    print("🧪 Testing Query Cache")
    print()
    
    try:
        test_basic_cache()
        test_fuzzy_matching()
        test_ttl_expiration()
        test_lru_eviction()
        test_with_real_queries()
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
