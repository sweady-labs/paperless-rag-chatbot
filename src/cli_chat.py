#!/usr/bin/env python3
"""
Fast CLI interface for the Paperless RAG Chatbot.
"""
import requests
import sys
import time

API_URL = "http://localhost:8001"


def chat():
    """Interactive chat loop."""
    print("=" * 50)
    print("Paperless RAG Chatbot (Fast Mode)")
    print("Type 'exit' or 'quit' to stop")
    print("=" * 50)
    print()
    
    # Check server health
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        data = health.json()
        
        if data.get('status') == 'healthy':
            collection = data.get('collection', {})
            config = data.get('config', {})
            print(f"Connected to server")
            print(f"  Chunks: {collection.get('points_count', 0)}")
            print(f"  LLM: {config.get('llm_model', 'unknown')}")
            print()
        else:
            print("Server running but may have issues")
            print()
            
    except requests.exceptions.RequestException:
        print(f"Cannot connect to server at {API_URL}")
        print("Run: make serve-api")
        sys.exit(1)
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            # Query with timing
            start = time.time()
            print("\nSearching...", end='', flush=True)
            
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "n_results": 3},
                timeout=120  # 120s timeout for long responses
            )
            
            latency = time.time() - start
            print("\r" + " " * 20 + "\r", end='')
            
            if response.status_code == 200:
                result = response.json()
                
                # Print answer
                print(f"Assistant: {result['answer']}")
                
                # Print sources
                if result['sources']:
                    print("\nSources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        score = source.get('score', 0) * 100
                        title = source.get('title', 'Unknown')
                        url = source.get('url', '')
                        
                        print(f"  {i}. {title} ({score:.0f}%)")
                        if url:
                            print(f"     {url}")
                
                # Show timing
                server_ms = result.get('latency_ms', 0)
                print(f"\n[{latency:.1f}s total, {server_ms}ms server]")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except requests.exceptions.Timeout:
            print("\rTimeout. Check if Ollama is running.")
        except Exception as e:
            print(f"\rError: {e}")


if __name__ == "__main__":
    chat()
