#!/usr/bin/env python3
"""
Simple CLI interface for the Paperless RAG Chatbot
"""
import requests
import json
import sys

API_URL = "http://localhost:8001"

def chat():
    """Interactive chat loop"""
    print("=" * 60)
    print("Paperless RAG Chatbot - Type 'exit' or 'quit' to stop")
    print("=" * 60)
    print()
    
    # Check if server is running
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        health_data = health.json()
        if health_data.get('status') == 'healthy':
            collection_info = health_data.get('collection', {})
            print(f"✓ Connected to server")
            print(f"✓ Indexed chunks: {collection_info.get('points_count', 'unknown')}")
            print()
        else:
            print("⚠ Server is running but vector store might not be ready")
            print()
    except requests.exceptions.RequestException:
        print("✗ Error: Cannot connect to server at", API_URL)
        print("  Make sure the server is running: python src/api/server.py")
        sys.exit(1)
    
    while True:
        try:
            question = input("\n🧑 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Show thinking indicator
            print("\n🤖 Assistant: Searching documents...", end='', flush=True)
            
            # Query the API
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "n_results": 5},
                timeout=60
            )
            
            # Clear the thinking indicator
            print("\r" + " " * 50 + "\r", end='')
            
            if response.status_code == 200:
                result = response.json()
                
                # Print answer
                print(f"🤖 Assistant: {result['answer']}\n")
                
                # Print sources
                if result['sources']:
                    print("📚 Sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        score_percent = source['score'] * 100
                        # Include URL if available
                        if 'url' in source and source['url']:
                            print(f"  {i}. {source['title']} (relevance: {score_percent:.1f}%)")
                            print(f"     → {source['url']}")
                        else:
                            print(f"  {i}. {source['title']} (relevance: {score_percent:.1f}%)")
                else:
                    print("📚 No sources found")
            else:
                print(f"✗ Error: {response.status_code} - {response.text}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except requests.exceptions.Timeout:
            print("\r✗ Request timed out. Try asking a simpler question.")
        except Exception as e:
            print(f"\r✗ Error: {e}")

if __name__ == "__main__":
    chat()
