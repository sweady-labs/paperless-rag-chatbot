# Paperless RAG Chatbot Makefile
# Simple commands for indexing and running the chatbot

.PHONY: help setup index serve-api serve-web serve-all stop clean

# Configuration
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
UVICORN := $(VENV_DIR)/bin/uvicorn

# Default target: show help
help:
	@echo "Paperless RAG Chatbot - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo ""
	@echo "Indexing (GPU-accelerated on M5):"
	@echo "  make index        - Index all documents from Paperless-NGX"
	@echo ""
	@echo "Serving:"
	@echo "  make serve-api    - Start API server on port 8001"
	@echo "  make serve-web    - Start web interface on port 7860"
	@echo "  make serve-all    - Start both API and web interface"
	@echo ""
	@echo "Maintenance:"
	@echo "  make stop         - Stop all running services"
	@echo "  make clean        - Remove virtual environment and cache"
	@echo "  make clean-data   - Remove vector database (preserves venv)"
	@echo ""

# Create virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	@if [ ! -d "$(VENV_DIR)" ]; then \
		python3 -m venv $(VENV_DIR); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists."; \
	fi
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Configure .env file (copy from .env.example if needed)"
	@echo "2. Run 'make index' to index your documents"
	@echo "3. Run 'make serve-all' to start the chatbot"

# Index all documents from Paperless-NGX (with GPU acceleration)
index:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting indexing with GPU acceleration (Metal/MPS)..."
	@$(PYTHON) -m src.indexer
	@echo "Indexing complete!"

# Start API server
serve-api:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting API server on http://localhost:8001"
	@echo "API docs available at http://localhost:8001/docs"
	@$(UVICORN) src.api.server:app --host 0.0.0.0 --port 8001

# Start web interface
serve-web:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting web interface on http://localhost:7860"
	@$(PYTHON) src/web_interface.py

# Start both API and web interface (in background)
serve-all:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@echo "Starting API server in background..."
	@$(UVICORN) src.api.server:app --host 0.0.0.0 --port 8001 > /dev/null 2>&1 & echo $$! > .api.pid
	@sleep 2
	@echo "Starting web interface..."
	@echo ""
	@echo "Services running:"
	@echo "  - API: http://localhost:8001 (docs at /docs)"
	@echo "  - Web: http://localhost:7860"
	@echo ""
	@echo "Run 'make stop' to stop all services"
	@$(PYTHON) src/web_interface.py

# Stop all running services
stop:
	@echo "Stopping services..."
	@if [ -f .api.pid ]; then \
		kill `cat .api.pid` 2>/dev/null || true; \
		rm .api.pid; \
		echo "API server stopped."; \
	fi
	@pkill -f "uvicorn src.api.server" 2>/dev/null || true
	@pkill -f "python.*web_interface" 2>/dev/null || true
	@pkill -f "gradio" 2>/dev/null || true
	@echo "All services stopped."

# Remove virtual environment and cache
clean:
	@echo "Removing virtual environment and cache..."
	@rm -rf $(VENV_DIR)
	@rm -rf cache/
	@rm -rf __pycache__ src/__pycache__ src/api/__pycache__ src/rag/__pycache__ src/utils/__pycache__
	@rm -f .api.pid
	@echo "Clean complete!"

# Remove vector database only (preserves venv and cache)
clean-data:
	@echo "Removing vector database..."
	@rm -rf data/vector_db/
	@echo "Vector database removed. Run 'make index' to re-index."
