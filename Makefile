# Paperless RAG Chatbot Makefile
# Optimized for Apple Silicon with Qdrant Docker

.PHONY: help setup setup-models index serve-all stop clean clean-all clean-logs qdrant-start qdrant-stop qdrant-status

# Configuration
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
UVICORN := $(VENV_DIR)/bin/uvicorn

# Default target: show help
help:
	@echo "========================================"
	@echo "Paperless RAG Chatbot"
	@echo "========================================"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup        - Install dependencies"
	@echo "  make setup-models - Download Ollama models"
	@echo "  make qdrant-start - Start Qdrant Docker"
	@echo "  make index        - Index all documents"
	@echo "  make serve-all    - Start API + Web UI"
	@echo ""
	@echo "Daily Use:"
	@echo "  make serve-all    - Start everything"
	@echo "  make stop         - Stop all services"
	@echo ""
	@echo "Qdrant Docker:"
	@echo "  make qdrant-start  - Start Qdrant container"
	@echo "  make qdrant-stop   - Stop Qdrant container"
	@echo "  make qdrant-status - Check Qdrant status"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove vector DB only"
	@echo "  make clean-logs   - Remove log files"
	@echo "  make clean-all    - Remove everything (reset)"
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
	@echo "✓ Setup complete! Next steps:"
	@echo "  1. make setup-models (download LLM models)"
	@echo "  2. Configure .env file"
	@echo "  3. make qdrant-start (start Docker)"
	@echo "  4. make index (index documents)"
	@echo "  5. make serve-all (start chatbot)"

# Download required Ollama models
setup-models:
	@echo "========================================"
	@echo "Downloading Ollama models..."
	@echo "========================================"
	@echo ""
	@echo "1. LLM Model: llama3.1:8b (~4.7 GB)"
	@ollama pull llama3.1:8b
	@echo ""
	@echo "2. Embedding Model: bge-m3 (~2.2 GB)"
	@ollama pull bge-m3
	@echo ""
	@echo "✓ Models ready!"
	@echo ""
	@echo "Note: These match your .env configuration."
	@echo "To use different models, update OLLAMA_MODEL and OLLAMA_EMBEDDING_MODEL in .env"

# Index all documents
index:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "========================================"
	@echo "Indexing documents from Paperless..."
	@echo "========================================"
	@$(PYTHON) -m src.indexer
	@echo ""
	@echo "✓ Indexing complete!"

# Start both API and web interface
serve-all:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Error: Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting API server in background..."
	@$(UVICORN) src.api.server:app --host 0.0.0.0 --port 8001 > /dev/null 2>&1 & echo $$! > .api.pid
	@sleep 3
	@echo ""
	@echo "========================================"
	@echo "Services Running:"
	@echo "  API:  http://localhost:8001"
	@echo "  Docs: http://localhost:8001/docs"
	@echo "  Web:  http://localhost:7860"
	@echo "========================================"
	@echo ""
	@echo "Press Ctrl+C to stop, or run 'make stop'"
	@echo ""
	@$(PYTHON) src/web_interface_enhanced.py

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
	@echo "✓ All services stopped."

# Remove vector database only
clean:
	@echo "Removing vector databases..."
	@rm -rf data/vector_db/
	@rm -rf data/qdrant_storage/
	@echo ""
	@echo "✓ Vector databases removed."
	@echo ""
	@echo "Note: If using Qdrant Docker, also run:"
	@echo "  docker-compose down -v"
	@echo ""
	@echo "To re-index: make index"

# Remove log files
clean-logs:
	@echo "Removing log files..."
	@rm -rf logs/
	@mkdir -p logs
	@echo "✓ Logs cleared."

# Remove everything (complete reset)
clean-all:
	@echo "WARNING: This will remove ALL data!"
	@echo "  - Virtual environment"
	@echo "  - Vector databases"
	@echo "  - Cache files"
	@echo "  - Log files"
	@echo "  - Metrics database"
	@echo ""
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "Removing all data..."
	@rm -rf $(VENV_DIR)
	@rm -rf data/
	@rm -rf cache/
	@rm -rf logs/
	@rm -rf __pycache__ src/__pycache__ src/api/__pycache__ src/rag/__pycache__ src/monitoring/__pycache__ src/utils/__pycache__
	@rm -f .api.pid
	@echo ""
	@echo "✓ Complete reset done!"
	@echo ""
	@echo "To start fresh:"
	@echo "  1. make setup"
	@echo "  2. make setup-models"
	@echo "  3. make qdrant-start"
	@echo "  4. make index"
	@echo "  5. make serve-all"

# =============================================================================
# Qdrant Docker Commands
# =============================================================================

# Start Qdrant Docker container
qdrant-start:
	@echo "Starting Qdrant Docker container..."
	@docker-compose up -d
	@echo ""
	@echo "Waiting for Qdrant to start..."
	@sleep 3
	@echo ""
	@docker ps | grep qdrant && echo "✓ Qdrant is running" || echo "⚠️  Qdrant not running"
	@echo ""
	@echo "Qdrant Dashboard: http://localhost:6333/dashboard"
	@echo "API Endpoint:     http://localhost:6333"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Ensure .env has: QDRANT_MODE=docker"
	@echo "  2. Run 'make index' to populate database"

# Stop Qdrant Docker container
qdrant-stop:
	@echo "Stopping Qdrant Docker container..."
	@docker-compose down
	@echo "✓ Qdrant stopped."

# Check Qdrant status
qdrant-status:
	@echo "========================================"
	@echo "Qdrant Status"
	@echo "========================================"
	@echo ""
	@echo "Docker Container:"
	@docker ps | grep qdrant && echo "  ✓ Running" || echo "  ✗ Not running (run 'make qdrant-start')"
	@echo ""
	@echo "API Health:"
	@curl -s http://localhost:6333/ > /dev/null 2>&1 && echo "  ✓ Responding" || echo "  ✗ Not responding"
	@echo ""
	@echo "Collections:"
	@curl -s http://localhost:6333/collections 2>/dev/null | grep -o '"name":"[^"]*"' || echo "  (Empty - run 'make index')"
	@echo ""
