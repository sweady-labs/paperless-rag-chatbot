# Paperless RAG Chatbot Makefile (Fast Mode)
# Optimized for Apple Silicon (M5) with sub-2-second responses

.PHONY: help setup setup-models index serve-api serve-web serve-all stop clean clean-data benchmark

# Configuration
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
UVICORN := $(VENV_DIR)/bin/uvicorn

# Default target: show help
help:
	@echo "========================================"
	@echo "Paperless RAG Chatbot (Fast Mode)"
	@echo "========================================"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup        - Install Python dependencies"
	@echo "  make setup-models - Download Ollama models (required)"
	@echo "  make index        - Index all documents"
	@echo "  make serve-all    - Start API + Web interface"
	@echo ""
	@echo "Individual Commands:"
	@echo "  make serve-api    - Start API server (port 8001)"
	@echo "  make serve-web    - Start web interface (port 7860)"
	@echo "  make stop         - Stop all services"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean-data   - Clear vector database"
	@echo "  make clean        - Remove everything (venv + data)"
	@echo "  make benchmark    - Test model speed"
	@echo ""

# Create virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment exists."; \
	fi
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip -q
	@$(PIP) install -r requirements.txt -q
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "  1. Run 'make setup-models' to download Ollama models"
	@echo "  2. Configure .env file"
	@echo "  3. Run 'make index' to index documents"
	@echo "  4. Run 'make serve-all' to start"

# Download required Ollama models
setup-models:
	@echo "========================================"
	@echo "Downloading Ollama models..."
	@echo "========================================"
	@echo ""
	@echo "1. LLM Model: qwen2.5:3b (~2GB)"
	@ollama pull qwen2.5:3b
	@echo ""
	@echo "2. Embedding Model: mxbai-embed-large (~670MB)"
	@ollama pull mxbai-embed-large
	@echo ""
	@echo "Models ready!"
	@echo ""
	@echo "Optional: For better quality (but slower), also run:"
	@echo "  ollama pull qwen2.5:7b"
	@echo "  ollama pull llama3.2:3b"

# Index all documents
index:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "========================================"
	@echo "Indexing documents..."
	@echo "========================================"
	@$(PYTHON) -m src.indexer
	@echo ""
	@echo "Indexing complete!"

# Start API server
serve-api:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting API server on http://localhost:8001"
	@echo "API docs: http://localhost:8001/docs"
	@$(UVICORN) src.api.server:app --host 0.0.0.0 --port 8001

# Start web interface
serve-web:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting web interface on http://localhost:7860"
	@$(PYTHON) src/web_interface.py

# Start both API and web interface
serve-all:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting API server in background..."
	@$(UVICORN) src.api.server:app --host 0.0.0.0 --port 8001 > /dev/null 2>&1 & echo $$! > .api.pid
	@sleep 3
	@echo ""
	@echo "========================================"
	@echo "Services Running:"
	@echo "  API:  http://localhost:8001 (docs at /docs)"
	@echo "  Web:  http://localhost:7860"
	@echo "========================================"
	@echo ""
	@echo "Run 'make stop' to stop all services"
	@echo ""
	@$(PYTHON) src/web_interface.py

# Stop all running services
stop:
	@echo "Stopping services..."
	@if [ -f .api.pid ]; then \
		kill `cat .api.pid` 2>/dev/null || true; \
		rm .api.pid; \
	fi
	@pkill -f "uvicorn src.api.server" 2>/dev/null || true
	@pkill -f "python.*web_interface" 2>/dev/null || true
	@pkill -f "gradio" 2>/dev/null || true
	@echo "All services stopped."

# Benchmark model speed
benchmark:
	@echo "========================================"
	@echo "Benchmarking Ollama models..."
	@echo "========================================"
	@echo ""
	@echo "Testing qwen2.5:3b..."
	@echo "What is 2+2?" | ollama run qwen2.5:3b --verbose 2>&1 | grep -E "eval rate|total duration"
	@echo ""
	@echo "Testing nomic-embed-text..."
	@curl -s http://localhost:11434/api/embeddings -d '{"model": "nomic-embed-text", "prompt": "Hello world"}' | head -c 100
	@echo ""
	@echo ""
	@echo "Benchmark complete!"

# Remove vector database only
clean-data:
	@echo "Removing vector database..."
	@rm -rf data/vector_db/
	@echo "Vector database removed."
	@echo "Run 'make index' to re-index."

# Remove everything
clean:
	@echo "Removing all data..."
	@rm -rf $(VENV_DIR)
	@rm -rf data/vector_db/
	@rm -rf cache/
	@rm -rf __pycache__ src/__pycache__ src/api/__pycache__ src/rag/__pycache__ src/utils/__pycache__
	@rm -f .api.pid
	@echo "Clean complete!"
