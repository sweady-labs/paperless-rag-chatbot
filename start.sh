#!/bin/bash

# Start Script for Paperless RAG Chatbot

set -e

echo "🚀 Starting Paperless RAG Chatbot..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run 'python3 -m venv venv' first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copy .env.example and configure it."
    exit 1
fi

# Check if Ollama is running
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running. Start it with 'ollama serve'"
    exit 1
fi

# Check if required models are available
echo "Checking Ollama models..."
if ! ollama list | grep -q "llama3.1:8b"; then
    echo "⚠️  llama3.1:8b not found. Pulling..."
    ollama pull llama3.1:8b
fi

if ! ollama list | grep -q "bge-m3"; then
    echo "⚠️  bge-m3 not found. Pulling..."
    ollama pull bge-m3
fi

echo "✅ All models ready"
echo ""

# Main menu
echo "Select what to run:"
echo "1) Index documents (first time or update)"
echo "2) Start API server (OPTIMIZED)"
echo "3) Start Web Interface"
echo "4) Start CLI Chat"
echo "5) Debug: Test query"
echo ""
read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        echo "Starting indexer..."
        python3 -m src.indexer
        ;;
    2)
        echo "Starting API server on http://localhost:8001 ..."
        python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001
        ;;
    3)
        echo "Starting Web Interface on http://localhost:7860 ..."
        echo "Make sure API server is running (Option 2)!"
        python3 src/web_interface.py
        ;;
    4)
        echo "Starting CLI Chat..."
        python3 -m src.cli_chat
        ;;
    5)
        echo ""
        read -p "Enter test query: " test_query
        echo "Testing search..."
        python3 -c "
from src.rag.hybrid_vector_store import HybridVectorStore
from src.rag.query_engine import QueryEngine
import time

vs = HybridVectorStore()
qe = QueryEngine(vs, use_reranker=True)

print(f'\\n📝 Query: {repr('$test_query')}')
print('=' * 60)

start = time.time()
result = qe.query('$test_query', n_results=5)
elapsed = time.time() - start

print(f'\\n✅ Answer (took {elapsed:.2f}s):')
print(result['answer'])
print(f'\\n📚 Sources ({len(result[\"sources\"])}):')
for i, s in enumerate(result['sources'], 1):
    print(f'{i}. {s[\"title\"]} (score: {s.get(\"score\", 0):.3f})')
"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
