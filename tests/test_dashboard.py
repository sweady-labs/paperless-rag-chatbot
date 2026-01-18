"""
Test the enhanced monitoring dashboard.

This script:
1. Starts the API server (in background)
2. Runs some test queries
3. Shows how to access metrics endpoints
"""

import requests
import time
import json

API_URL = "http://localhost:8001"

def test_metrics_endpoints():
    """Test all metrics endpoints."""
    print("=" * 60)
    print("TESTING METRICS ENDPOINTS")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            data = response.json()
            print(f"   Chunks: {data['collection']['points_count']}")
            print(f"   LLM: {data['config']['llm_model']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Make sure API server is running:")
        print("   cd /Users/enricoschmidt/paperless-rag-chatbot")
        print("   source .venv/bin/activate")
        print("   uvicorn src.api.server:app --host 0.0.0.0 --port 8001")
        return
    
    # Test 2: Live metrics
    print("\n2. Testing /metrics/live endpoint...")
    try:
        response = requests.get(f"{API_URL}/metrics/live", timeout=5)
        if response.status_code == 200:
            print("✅ Live metrics available")
            data = response.json()
            stats = data.get('statistics', {})
            print(f"   Total queries: {stats.get('total_queries', 0)}")
            if stats.get('total_queries', 0) > 0:
                latency = stats.get('latency', {})
                print(f"   Avg latency: {latency.get('mean', 0):.1f}ms")
        else:
            print(f"❌ Live metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Recent queries
    print("\n3. Testing /metrics/recent endpoint...")
    try:
        response = requests.get(f"{API_URL}/metrics/recent?limit=5", timeout=5)
        if response.status_code == 200:
            print("✅ Recent queries available")
            data = response.json()
            queries = data.get('queries', [])
            print(f"   Found {len(queries)} recent queries")
            for i, q in enumerate(queries[:3], 1):
                print(f"   {i}. \"{q['query_text'][:50]}...\" ({q['total_duration_ms']:.0f}ms)")
        else:
            print(f"❌ Recent queries failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: 24h statistics
    print("\n4. Testing /metrics/statistics endpoint...")
    try:
        response = requests.get(f"{API_URL}/metrics/statistics", timeout=5)
        if response.status_code == 200:
            print("✅ Statistics available")
            data = response.json()
            db_stats = data.get('last_24h', {})
            mem_stats = data.get('in_memory', {})
            print(f"   In-memory queries: {mem_stats.get('total_queries', 0)}")
            print(f"   Last 24h queries: {db_stats.get('total_queries', 0)}")
        else:
            print(f"❌ Statistics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Slow queries
    print("\n5. Testing /metrics/slow endpoint...")
    try:
        response = requests.get(f"{API_URL}/metrics/slow?threshold_ms=3000&limit=5", timeout=5)
        if response.status_code == 200:
            print("✅ Slow queries endpoint working")
            data = response.json()
            queries = data.get('queries', [])
            print(f"   Found {len(queries)} slow queries (>3000ms)")
            for i, q in enumerate(queries[:3], 1):
                print(f"   {i}. \"{q['query_text'][:50]}...\" ({q['total_duration_ms']:.0f}ms)")
        else:
            print(f"❌ Slow queries failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ METRICS ENDPOINTS TEST COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the enhanced web interface:")
    print("   python src/web_interface_enhanced.py")
    print()
    print("2. Open in browser:")
    print("   http://localhost:7860")
    print()
    print("3. Try some queries and watch the metrics update!")


if __name__ == "__main__":
    test_metrics_endpoints()
