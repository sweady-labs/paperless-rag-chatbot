#!/usr/bin/env python3
"""
Test streaming query functionality.

Tests both the query engine directly and via API endpoint.
"""

import sys
import json
import time
import requests

def test_direct_streaming():
    """Test streaming directly from query engine."""
    print("=" * 60)
    print("Test 1: Direct Streaming from Query Engine")
    print("=" * 60)
    print()
    
    from src.rag.fast_vector_store import FastVectorStore
    from src.rag.fast_query_engine import FastQueryEngine
    
    # Initialize
    vector_store = FastVectorStore()
    query_engine = FastQueryEngine(vector_store)
    
    query = "Wann ist Mavi geboren?"
    print(f"Query: {query}")
    print()
    print("Streaming answer:")
    print("-" * 60)
    
    start_time = time.time()
    metadata = None
    
    for chunk in query_engine.stream_query(query):
        if chunk['type'] == 'token':
            # Print tokens as they arrive
            print(chunk['data'], end='', flush=True)
        elif chunk['type'] == 'metadata':
            # Save metadata for final output
            metadata = chunk['data']
    
    duration = (time.time() - start_time) * 1000
    
    print()
    print("-" * 60)
    print()
    
    if metadata:
        print(f"✓ Streaming complete in {duration:.0f}ms")
        print(f"Sources: {metadata['metrics']['num_sources']}")
        print(f"Top score: {metadata['metrics']['top_score']}")
        print(f"LLM speed: {metadata['metrics']['tokens_per_sec']:.1f} tok/s")
        print()
        return True
    else:
        print("✗ No metadata received")
        return False


def test_api_streaming():
    """Test streaming via API endpoint."""
    print("=" * 60)
    print("Test 2: Streaming via API Endpoint")
    print("=" * 60)
    print()
    
    API_URL = "http://localhost:8001"
    
    # Check API is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code != 200:
            print("✗ API not healthy")
            return False
    except Exception as e:
        print(f"✗ API not running: {e}")
        print("Start API with: make serve-api")
        return False
    
    query = "Lohnsteuerbescheinigung 2024"
    print(f"Query: {query}")
    print()
    print("Streaming answer:")
    print("-" * 60)
    
    start_time = time.time()
    metadata = None
    
    try:
        # Stream request
        response = requests.post(
            f"{API_URL}/query/stream",
            json={"question": query},
            stream=True,
            timeout=30
        )
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                
                if chunk['type'] == 'token':
                    print(chunk['data'], end='', flush=True)
                elif chunk['type'] == 'metadata':
                    metadata = chunk['data']
        
        duration = (time.time() - start_time) * 1000
        
        print()
        print("-" * 60)
        print()
        
        if metadata:
            print(f"✓ Streaming complete in {duration:.0f}ms")
            print(f"Sources: {metadata['metrics']['num_sources']}")
            print(f"Top score: {metadata['metrics']['top_score']}")
            print(f"Total latency: {metadata['metrics']['total_duration_ms']:.0f}ms")
            print()
            return True
        else:
            print("✗ No metadata received")
            return False
            
    except Exception as e:
        print(f"\n✗ Streaming error: {e}")
        return False


def main():
    """Run all tests."""
    print()
    print("🚀 Testing Streaming Functionality")
    print()
    
    results = {}
    
    # Test 1: Direct streaming
    try:
        results['direct'] = test_direct_streaming()
    except Exception as e:
        print(f"✗ Direct streaming failed: {e}")
        results['direct'] = False
    
    # Test 2: API streaming
    try:
        results['api'] = test_api_streaming()
    except Exception as e:
        print(f"✗ API streaming failed: {e}")
        results['api'] = False
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Direct Streaming: {'✓ PASS' if results.get('direct') else '✗ FAIL'}")
    print(f"API Streaming:    {'✓ PASS' if results.get('api') else '✗ FAIL'}")
    print()
    
    if all(results.values()):
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
