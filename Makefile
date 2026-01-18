# Paperless RAG Chatbot Makefile
# Optimized for Apple Silicon with Qdrant Docker

.PHONY: help help-verbose setup setup-models index title-preview title-preview-all title-apply serve-all stop clean clean-all clean-logs qdrant-start qdrant-stop qdrant-status

# Configuration
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
UVICORN := $(VENV_DIR)/bin/uvicorn

# Colors for terminal output
YELLOW := \033[1;33m
WHITE := \033[0;37m
GRAY := \033[0;90m
RESET := \033[0m
BOLD := \033[1m

# Default target: show help
help:
	@echo "========================================"
	@echo "$(BOLD)Paperless RAG Chatbot$(RESET)"
	@echo "========================================"
	@echo ""
	@echo "$(BOLD)Quick Start:$(RESET)"
	@printf "  $(YELLOW)make setup$(RESET)        $(GRAY)- Install dependencies$(RESET)\n"
	@printf "  $(YELLOW)make setup-models$(RESET) $(GRAY)- Download Ollama models$(RESET)\n"
	@printf "  $(YELLOW)make qdrant-start$(RESET) $(GRAY)- Start Qdrant Docker$(RESET)\n"
	@printf "  $(YELLOW)make index$(RESET)        $(GRAY)- Index all documents$(RESET)\n"
	@printf "  $(YELLOW)make serve-all$(RESET)    $(GRAY)- Start API + Web UI$(RESET)\n"
	@echo ""
	@echo "$(BOLD)Daily Use:$(RESET)"
	@printf "  $(YELLOW)make serve-all$(RESET)    $(GRAY)- Start everything$(RESET)\n"
	@printf "  $(YELLOW)make stop$(RESET)         $(GRAY)- Stop all services$(RESET)\n"
	@echo ""
	@echo "$(BOLD)Document Management:$(RESET)"
	@printf "  $(YELLOW)make title-preview$(RESET)     $(GRAY)- Preview auto-titling (10 docs, safe)$(RESET)\n"
	@printf "  $(YELLOW)make title-preview-all$(RESET) $(GRAY)- Preview all title changes (safe)$(RESET)\n"
	@printf "  $(YELLOW)make title-apply$(RESET)       $(GRAY)- Apply auto-titling to ALL docs (~16min)$(RESET)\n"
	@echo ""
	@echo "$(BOLD)Qdrant Docker:$(RESET)"
	@printf "  $(YELLOW)make qdrant-start$(RESET)  $(GRAY)- Start Qdrant container$(RESET)\n"
	@printf "  $(YELLOW)make qdrant-stop$(RESET)   $(GRAY)- Stop Qdrant container$(RESET)\n"
	@printf "  $(YELLOW)make qdrant-status$(RESET) $(GRAY)- Check Qdrant status$(RESET)\n"
	@echo ""
	@echo "$(BOLD)Maintenance:$(RESET)"
	@printf "  $(YELLOW)make clean$(RESET)        $(GRAY)- Remove vector DB only$(RESET)\n"
	@printf "  $(YELLOW)make clean-logs$(RESET)   $(GRAY)- Remove log files$(RESET)\n"
	@printf "  $(YELLOW)make clean-all$(RESET)    $(GRAY)- Remove everything (reset)$(RESET)\n"
	@echo ""
	@printf "$(GRAY)For detailed command info: $(YELLOW)make help-verbose$(RESET)\n"
	@echo ""

# Show detailed help with full command descriptions
help-verbose:
	@echo "========================================"
	@echo "$(BOLD)Paperless RAG Chatbot - Detailed Help$(RESET)"
	@echo "========================================"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make setup$(RESET)"
	@printf "  $(GRAY)Install Python dependencies and create virtual environment$(RESET)\n"
	@printf "  $(GRAY)First-time setup - only needs to be run once$(RESET)\n"
	@printf "  $(GRAY)Creates .venv/ directory and installs requirements.txt$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make setup-models$(RESET)"
	@printf "  $(GRAY)Download required Ollama models (llama3.1:8b + bge-m3)$(RESET)\n"
	@printf "  $(GRAY)Total download size: ~7GB (4.7GB LLM + 2.2GB embeddings)$(RESET)\n"
	@printf "  $(GRAY)Only needs to be run once (models are cached by Ollama)$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make index$(RESET)"
	@printf "  $(GRAY)Initial indexing of all documents from Paperless-NGX$(RESET)\n"
	@printf "  $(GRAY)Fetches all documents, creates embeddings, builds vector database$(RESET)\n"
	@printf "  $(GRAY)Run after setup or when starting fresh$(RESET)\n"
	@printf "  $(GRAY)Duration: ~10 minutes for 650 documents$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make title-preview$(RESET)"
	@printf "  $(GRAY)Preview auto-titling on first 10 documents (dry run - no changes)$(RESET)\n"
	@printf "  $(GRAY)Uses local qwen2.5:3b LLM to generate descriptive German titles$(RESET)\n"
	@printf "  $(GRAY)Safe to run anytime - only shows what WOULD be changed$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make title-preview-all$(RESET)"
	@printf "  $(GRAY)Preview auto-titling for ALL documents (dry run - no changes)$(RESET)\n"
	@printf "  $(GRAY)Shows what titles would be changed across entire Paperless database$(RESET)\n"
	@printf "  $(GRAY)Useful for reviewing all changes before committing$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make title-apply$(RESET)"
	@printf "  $(GRAY)Apply auto-titling to ALL documents (LIVE - makes real changes!)$(RESET)\n"
	@printf "  $(GRAY)WARNING: This updates titles in Paperless database$(RESET)\n"
	@printf "  $(GRAY)Takes ~16 minutes for 650 documents (~1.5s per doc)$(RESET)\n"
	@printf "  $(GRAY)After completion, run 'make index' to update RAG search$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make serve-all$(RESET)"
	@printf "  $(GRAY)Start API server (port 8001) and web UI (port 7860)$(RESET)\n"
	@printf "  $(GRAY)API runs in background, web UI runs in foreground$(RESET)\n"
	@printf "  $(GRAY)Press Ctrl+C to stop web UI, then run 'make stop' to cleanup$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make stop$(RESET)"
	@printf "  $(GRAY)Stop all running services (API server + web UI)$(RESET)\n"
	@printf "  $(GRAY)Kills background processes and cleans up PID files$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make clean$(RESET)"
	@printf "  $(GRAY)Remove vector database only (preserves Paperless data)$(RESET)\n"
	@printf "  $(GRAY)Use when you want to re-index from scratch$(RESET)\n"
	@printf "  $(GRAY)Run 'make index' after to rebuild database$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make clean-logs$(RESET)"
	@printf "  $(GRAY)Remove all log files and recreate empty logs/ directory$(RESET)\n"
	@printf "  $(GRAY)Useful for fresh start or when logs get too large$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make clean-all$(RESET)"
	@printf "  $(GRAY)COMPLETE RESET - removes everything except source code$(RESET)\n"
	@printf "  $(GRAY)Deletes: virtual env, vector DB, cache, logs, metrics$(RESET)\n"
	@printf "  $(GRAY)Requires confirmation prompt - cannot be undone$(RESET)\n"
	@printf "  $(GRAY)After running, must redo: make setup, make setup-models, etc.$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make qdrant-start$(RESET)"
	@printf "  $(GRAY)Start Qdrant vector database in Docker container$(RESET)\n"
	@printf "  $(GRAY)Starts in background (daemon mode) and waits for startup$(RESET)\n"
	@printf "  $(GRAY)Dashboard available at http://localhost:6333/dashboard$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make qdrant-stop$(RESET)"
	@printf "  $(GRAY)Stop Qdrant Docker container and remove volumes$(RESET)\n"
	@printf "  $(GRAY)Safe to run - data is preserved in data/qdrant_storage/$(RESET)\n"
	@echo ""
	@echo "$(BOLD)$(YELLOW)make qdrant-status$(RESET)"
	@printf "  $(GRAY)Check if Qdrant is running and show collection info$(RESET)\n"
	@printf "  $(GRAY)Shows: Docker status, API health, indexed collections$(RESET)\n"
	@echo ""

# Install Python dependencies and create virtual environment
# First-time setup - only needs to be run once
# Creates .venv/ directory and installs requirements.txt
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

# Download required Ollama models (llama3.1:8b + bge-m3)
# Total download size: ~7GB (4.7GB LLM + 2.2GB embeddings)
# Only needs to be run once (models are cached by Ollama)
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

# Initial indexing of all documents from Paperless-NGX
# Fetches all documents, creates embeddings, builds vector database
# Run after setup or when starting fresh
# Duration: ~10 minutes for 650 documents
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

# =============================================================================
# Document Auto-Titling (LLM-powered intelligent title generation)
# =============================================================================

# Preview auto-titling on first 10 documents (dry run - no changes)
# Uses local qwen2.5:3b LLM to generate descriptive German titles
# Safe to run anytime - only shows what WOULD be changed
title-preview:
	@echo "========================================"
	@echo "Preview Auto-Titling (10 Documents)"
	@echo "========================================"
	@echo ""
	@echo "This will preview LLM-generated titles for 10 documents."
	@echo "No changes will be made to Paperless."
	@echo ""
	@PYTHONPATH=$(PWD) $(PYTHON) scripts/auto_title_documents.py --all --limit=10
	@echo ""
	@echo "To preview all documents: make title-preview-all"
	@echo "To apply changes:         make title-apply"

# Preview auto-titling for ALL documents (dry run - no changes)
# Shows what titles would be changed across entire Paperless database
# Useful for reviewing all changes before committing
title-preview-all:
	@echo "========================================"
	@echo "Preview Auto-Titling (ALL Documents)"
	@echo "========================================"
	@echo ""
	@echo "This will preview LLM-generated titles for ALL documents."
	@echo "No changes will be made to Paperless."
	@echo ""
	@PYTHONPATH=$(PWD) $(PYTHON) scripts/auto_title_documents.py --all
	@echo ""
	@echo "To apply these changes: make title-apply"

# Apply auto-titling to ALL documents (LIVE - makes real changes!)
# WARNING: This updates titles in Paperless database
# Takes ~16 minutes for 650 documents (~1.5s per doc)
# After completion, run 'make index' to update RAG search
title-apply:
	@echo "========================================"
	@echo "⚠️  WARNING: LIVE MODE - REAL CHANGES"
	@echo "========================================"
	@echo ""
	@echo "This will AUTO-TITLE ALL documents in Paperless using LLM."
	@echo ""
	@echo "What it does:"
	@echo "  - Uses qwen2.5:3b (local LLM) to generate titles"
	@echo "  - Updates ~650 documents in Paperless database"
	@echo "  - Takes approximately 16 minutes"
	@echo "  - Changes are PERMANENT (updates Paperless DB)"
	@echo ""
	@echo "After completion, run 'make index' to update RAG search."
	@echo ""
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo ""
	@echo "Starting auto-titling..."
	@PYTHONPATH=$(PWD) $(PYTHON) scripts/auto_title_documents.py --all --apply
	@echo ""
	@echo "✓ Titles updated in Paperless!"
	@echo ""
	@echo "⚠️  IMPORTANT: Run 'make index' to update RAG search database."

# Start API server (port 8001) and web UI (port 7860)
# API runs in background, web UI runs in foreground
# Press Ctrl+C to stop web UI, then run 'make stop' to cleanup
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

# Stop all running services (API server + web UI)
# Kills background processes and cleans up PID files
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

# Remove vector database only (preserves Paperless data)
# Use when you want to re-index from scratch
# Run 'make index' or 'make reindex' after to rebuild database
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

# Remove all log files and recreate empty logs/ directory
# Useful for fresh start or when logs get too large
clean-logs:
	@echo "Removing log files..."
	@rm -rf logs/
	@mkdir -p logs
	@echo "✓ Logs cleared."

# COMPLETE RESET - removes everything except source code
# Deletes: virtual env, vector DB, cache, logs, metrics
# Requires confirmation prompt - cannot be undone
# After running, must redo: make setup, make setup-models, etc.
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

# Start Qdrant vector database in Docker container
# Starts in background (daemon mode) and waits for startup
# Dashboard available at http://localhost:6333/dashboard
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

# Stop Qdrant Docker container and remove volumes
# Safe to run - data is preserved in data/qdrant_storage/
qdrant-stop:
	@echo "Stopping Qdrant Docker container..."
	@docker-compose down
	@echo "✓ Qdrant stopped."

# Check if Qdrant is running and show collection info
# Shows: Docker status, API health, indexed collections
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
