# Paperless RAG Chatbot

AI-powered document search for Paperless-NGX using local LLMs and vector embeddings. Optimized for Apple Silicon.

## Features

- **Hybrid Search** - Combines vector embeddings (60%) and BM25 keyword search (40%)
- **German Synonym Expansion** - Automatically expands queries with synonyms
- **Query Cache** - 800x faster responses for repeated queries
- **Metal GPU Acceleration** - Native Ollama integration for M-series Macs
- **LLM Streaming** - Real-time response generation
- **Monitoring Dashboard** - Query metrics and performance tracking

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  MacBook M5 (Host)                                                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Ollama (Native macOS)                                      │  │
│  │  ├─ LLM: llama3.1:8b (20-30 tok/s)                          │  │
│  │  ├─ Embeddings: bge-m3 (100-150ms)                          │  │ 
│  │  └─ Metal GPU acceleration ⚡                                │  │
│  │     Port: 11434                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ▲                                    │
│                              │ HTTP                               │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Python Application (Native)                                │  │
│  │  ├─ FastAPI Server (port 8001)                              │  │
│  │  │  └─ REST API + Streaming endpoints                       │  │
│  │  ├─ Gradio Web UI (port 7860)                               │  │
│  │  │  └─ Chat interface + Monitoring dashboard                │  │
│  │  └─ RAG Engine                                              │  │
│  │     ├─ Query cache (LRU + TTL)                              │  │
│  │     ├─ Synonym expansion (German)                           │  │
│  │     └─ Hybrid retriever (Vector 60% + BM25 40%)             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ▲                                    │
│                              │ HTTP                               │
│                              ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Qdrant (Docker Container)                                  │  │
│  │  ├─ Vector database (HNSW index)                            │  │
│  │  ├─ chunks from documents                                   │  │
│  │  └─ Storage: ./data/qdrant_storage                          │  │
│  │     Ports: 6333 (HTTP), 6334 (gRPC)                         │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  External:                                                        │
│  └─ Paperless-NGX (<yourIP>:8000)                                 │
│     └─ Source of documents                                        │
└───────────────────────────────────────────────────────────────────┘
```

**Design Decision:** Ollama runs natively (not containerized) because Docker on macOS cannot access Metal GPU. Native execution provides **10x faster** inference and embeddings.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/)
- Docker Desktop
- Paperless-NGX instance with API token

### Installation

```bash
# Clone and setup
git clone https://github.com/sweady-labs/paperless-rag-chatbot.git
cd paperless-rag-chatbot
make setup

# Download models (~7 GB)
make setup-models

# Start Qdrant
make qdrant-start

# Configure
cp .env.example .env
nano .env  # Set PAPERLESS_URL and PAPERLESS_TOKEN
```

### Usage

```bash
# Index documents
make index

# Start chatbot
make serve-all
```

- **Web UI**: http://localhost:7860
- **API**: http://localhost:8001/docs
- **Qdrant**: http://localhost:6333/dashboard

## Commands

```bash
# Setup
make setup          # Install dependencies
make setup-models   # Download llama3.1:8b + bge-m3
make qdrant-start   # Start Qdrant Docker

# Daily Use
make index          # Index documents from Paperless
make serve-all      # Start API + Web UI
make stop           # Stop all services

# Maintenance
make clean          # Remove vector database
make clean-logs     # Clear logs
make clean-all      # Complete reset
make qdrant-status  # Check Qdrant health
```

## Configuration

Key settings in `.env`:

```bash
# Paperless
PAPERLESS_URL=http://localhost:8000
PAPERLESS_TOKEN=your_token_here

# Ollama
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=bge-m3

# Qdrant
QDRANT_MODE=docker  # file or docker (docker recommended)

# Performance
ENABLE_CACHE=true
ENABLE_SYNONYM_EXPANSION=true
USE_HYBRID_SEARCH=true
```

## Project Structure

```
src/
├── api/
│   ├── paperless_client.py    # Paperless-NGX integration
│   └── server.py              # FastAPI server
├── monitoring/
│   ├── cache.py               # Query cache (LRU + TTL)
│   ├── metrics.py             # Performance metrics
│   └── logger.py              # JSON logging
├── rag/
│   ├── fast_vector_store.py   # Qdrant integration
│   ├── hybrid_retriever.py    # Vector + BM25 search
│   ├── fast_query_engine.py   # RAG engine
│   └── synonyms.py            # German synonym expansion
├── indexer.py                 # Document indexing
├── cli_chat.py                # CLI interface
└── web_interface_enhanced.py  # Gradio UI + dashboard

tests/                         # Test suite
scripts/                       # Utilities
data/                          # Vector DB + metrics
logs/                          # Query logs
```

## Performance

Tested on M5 MacBook with 651 documents:

| Operation | Time |
|-----------|------|
| First query | ~4.1s |
| Cached query | ~50ms |
| Indexing | 10-15 min |
| LLM speed | 20-30 tok/s |

## API Examples

```bash
# Query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Rechnungen 2024"}'

# Stream response
curl -X POST http://localhost:8001/query/stream \
  -d '{"query": "Arztbriefe"}' --no-buffer

# Cache stats
curl http://localhost:8001/cache/stats

# Health check
curl http://localhost:8001/health
```

## Troubleshooting

```bash
# Ollama not running
ollama serve
curl http://localhost:11434/api/tags

# Qdrant not running
make qdrant-start
make qdrant-status

# Test cache
python tests/test_cache.py

# Full system test
python tests/comprehensive_test.py
```

## License

MIT

## Acknowledgments

- [Paperless-NGX](https://github.com/paperless-ngx/paperless-ngx)
- [Ollama](https://ollama.ai/)
- [Qdrant](https://qdrant.tech/)
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding)
