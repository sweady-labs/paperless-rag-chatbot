#!/bin/bash
set -e

echo "🚀 Starting Paperless RAG Chatbot..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to check Ollama
check_ollama() {
    echo "🔍 Checking Ollama connection..."
    
    if ! curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
        echo "❌ ERROR: Ollama is not reachable at ${OLLAMA_BASE_URL}"
        echo "   Make sure Ollama is running on your host machine"
        echo "   Start it with: ollama serve"
        return 1
    fi
    
    echo "✅ Ollama is reachable"
    return 0
}

# Function to check if required model exists
check_model() {
    local model="${OLLAMA_MODEL:-gemma2:2b}"
    echo "🔍 Checking for model: $model"
    
    if curl -s "${OLLAMA_BASE_URL}/api/tags" | grep -q "\"name\":\"$model\""; then
        echo "✅ Model $model is available"
        return 0
    else
        echo "⚠️  WARNING: Model $model not found"
        echo "   Available models:"
        curl -s "${OLLAMA_BASE_URL}/api/tags" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | sed 's/^/   - /'
        echo ""
        echo "   To pull the model, run on your host:"
        echo "   ollama pull $model"
        return 1
    fi
}

# Function to check Paperless connection
check_paperless() {
    echo "🔍 Checking Paperless-NGX connection..."
    
    if ! curl -s -H "Authorization: Token ${PAPERLESS_TOKEN}" "${PAPERLESS_URL}/api/" > /dev/null 2>&1; then
        echo "❌ ERROR: Cannot connect to Paperless at ${PAPERLESS_URL}"
        echo "   Check your PAPERLESS_URL and PAPERLESS_TOKEN in .env"
        return 1
    fi
    
    # Get document count
    doc_count=$(curl -s -H "Authorization: Token ${PAPERLESS_TOKEN}" "${PAPERLESS_URL}/api/documents/?page_size=1" | grep -o '"count":[0-9]*' | cut -d':' -f2 || echo "0")
    echo "✅ Paperless is reachable ($doc_count documents)"
    return 0
}

# Function to check if indexing is needed
check_indexing() {
    echo "🔍 Checking vector database status..."
    
    if [ ! -f "/app/data/vector_db/meta.json" ]; then
        echo "⚠️  No vector database found - indexing required"
        return 1
    fi
    
    echo "✅ Vector database exists"
    return 0
}

# Function to run indexing
run_indexing() {
    echo ""
    echo "📚 Starting document indexing..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd /app
    python3 -m src.indexer
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Indexing complete!"
    echo ""
}

# Function to start services
start_services() {
    echo ""
    echo "🎯 Starting services..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd /app
    
    # Start API server in background
    echo "🚀 Starting API server on port 8001..."
    python3 -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001 &
    API_PID=$!
    
    # Wait for API to be ready
    echo "⏳ Waiting for API server to be ready..."
    max_wait=120  # 2 minutes max
    elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        if curl -s http://localhost:8001/health > /dev/null 2>&1; then
            echo "✅ API server is ready!"
            break
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $((elapsed % 10)) -eq 0 ]; then
            echo "   Still loading model... (${elapsed}s elapsed)"
        fi
    done
    
    if [ $elapsed -ge $max_wait ]; then
        echo "⚠️  WARNING: API server did not respond within ${max_wait}s"
        echo "   Continuing anyway..."
    fi
    
    # Start Web Interface in background
    echo "🌐 Starting Web Interface on port 7860..."
    python3 src/web_interface.py &
    WEB_PID=$!
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Services started successfully!"
    echo ""
    echo "📡 API Server:     http://localhost:8001"
    echo "   API Docs:       http://localhost:8001/docs"
    echo "🌐 Web Interface:  http://localhost:7860"
    echo ""
    echo "Press Ctrl+C to stop"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Wait for both processes
    wait $API_PID $WEB_PID
}

# Main execution
main() {
    # Step 1: Check Ollama
    if ! check_ollama; then
        echo ""
        echo "❌ Startup failed: Ollama not accessible"
        exit 1
    fi
    
    # Step 2: Check model
    if ! check_model; then
        echo ""
        echo "⚠️  Warning: Required model not found, but continuing..."
        echo "   You may experience errors until the model is available"
    fi
    
    # Step 3: Check Paperless
    if ! check_paperless; then
        echo ""
        echo "❌ Startup failed: Paperless not accessible"
        exit 1
    fi
    
    # Step 4: Check/run indexing
    if [ "${AUTO_INDEX}" = "true" ] || [ "${AUTO_INDEX}" = "1" ]; then
        echo ""
        echo "🔄 AUTO_INDEX is enabled - running indexing..."
        run_indexing
    else
        if ! check_indexing; then
            echo ""
            echo "⚠️  WARNING: No vector database found!"
            echo "   Set AUTO_INDEX=true in docker-compose.yml to index automatically"
            echo "   Or run manually: docker exec paperless-rag-chatbot python3 -m src.indexer"
            echo ""
            echo "   Continuing anyway..."
        fi
    fi
    
    # Step 5: Start services
    start_services
}

# Run main function
main
