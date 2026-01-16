# paperless-rag-chatbot

Small, local RAG chatbot for querying documents stored in Paperless‑NGX.

This project combines document chunking, BGE‑M3 embeddings, a Qdrant vector store and a local LLM via Ollama. It's designed to be small, practical and easy to run locally.
# Paperless RAG Chatbot

AI-powered chatbot for querying documents in Paperless-NGX using RAG (Retrieval-Augmented Generation) with **BGE-M3 hybrid search**.

## Features

- **GPU-Accelerated Indexing**: Uses Apple Metal (M5) for fast document processing
- **BGE-M3 Hybrid Search**: Multi-functionality retrieval (dense + sparse + ColBERT)
- **Multilingual Support**: Works with 100+ languages via BGE-M3
- **Long Document Support**: Up to 8192 tokens per chunk
- **Re-ranking**: Improved accuracy with fast keyword-based reranking
- **Local LLM**: Uses Ollama (gemma2:2b for fast responses)
- **Vector Database**: Qdrant for efficient similarity search
- **FastAPI Backend**: RESTful API for integration
- **Multiple Interfaces**: CLI and Web UI (Gradio)
- **Optimized Performance**: Fast queries with low CPU usage
- **Clickable Links**: Direct links to documents in Paperless

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Paperless-NGX instance with API access
- Paperless API token ([create one in Paperless settings](https://docs.paperless-ngx.com/api/#authorization))
- Apple Silicon Mac (for GPU acceleration) or any system with Python

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/sweady-labs/paperless-rag-chatbot.git
cd paperless-rag-chatbot

# Setup virtual environment and install dependencies
make setup
```

### 2. Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with your settings
```

Required settings in `.env`:
```bash
PAPERLESS_URL=http://your-paperless-url:8000
PAPERLESS_TOKEN=your_api_token_here
OLLAMA_MODEL=gemma2:2b
```

### 3. Install Ollama Model

```bash
# Install the lightweight model (recommended)
ollama pull gemma2:2b

# Or use a more powerful model for better quality
# ollama pull llama3.1:8b
# ollama pull qwen2.5:7b
```

Note: BGE-M3 embeddings are downloaded automatically on first use (~2.3GB)

### 4. Index Your Documents

```bash
# Index all documents with GPU acceleration (Apple Metal/MPS)
make index
```

### 5. Start the Chatbot

```bash
# Start both API server and web interface
make serve-all

# Or start components separately:
# make serve-api   # API server only (port 8001)
# make serve-web   # Web interface only (port 7860)
```

### 6. Use the Chatbot

- **Web Interface**: Open http://localhost:7860
- **API Documentation**: http://localhost:8001/docs
- **CLI Chat**: `python src/cli_chat.py`

## Available Commands

Run `make` or `make help` to see all available commands:

```bash
make setup        # Create virtual environment and install dependencies
make index        # Index all documents with GPU acceleration
make serve-api    # Start API server on port 8001
make serve-web    # Start web interface on port 7860
make serve-all    # Start both API and web interface
make stop         # Stop all running services
make clean        # Remove virtual environment and cache
make clean-data   # Remove vector database only
```

## Usage

### Web Interface

1. Open http://localhost:7860
2. Type your question in the chat box
3. Click on document links to open them in Paperless-NGX

### API

```bash
# Query documents
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me invoices from 2024"}'

# Check health
curl http://localhost:8001/health

# View API docs
open http://localhost:8001/docs
```

### API Endpoints

- `GET /` - API information
- `POST /query` - Query documents
- `GET /health` - Health check & stats
- `GET /stats` - Indexing statistics
- `GET /docs` - Interactive API documentation (Swagger)
- `POST /reindex` - Re-index all documents

## Project Structure

```
paperless-rag-chatbot/
├── .env.example            # Example configuration
├── requirements.txt        # Python dependencies
├── Makefile               # Build and run commands
├── src/
│   ├── api/
│   │   ├── paperless_client.py  # Paperless API integration
│   │   └── server.py            # FastAPI server
│   ├── rag/
│   │   ├── chunker.py           # Document chunking
│   │   ├── hybrid_vector_store.py  # BGE-M3 hybrid store
│   │   ├── query_engine.py      # RAG query logic
│   │   └── reranker.py          # Re-ranking for accuracy
│   ├── utils/
│   │   └── id_utils.py          # Stable ID generation
│   ├── indexer.py          # Document indexing script
│   ├── cli_chat.py         # CLI interface
│   ├── web_interface.py    # Gradio web UI
│   └── config.py           # Configuration management
└── data/
    └── vector_db/          # Qdrant storage
```

## Configuration

Edit `.env` to customize settings:

```bash
# Paperless Connection
PAPERLESS_URL=http://localhost:8000
PAPERLESS_TOKEN=your_api_token_here

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma2:2b        # Lightweight and fast

# Chunking (optimized for BGE-M3's 8K token window)
CHUNK_SIZE=2000               # Tokens per chunk
CHUNK_OVERLAP=300             # Overlap between chunks

# BGE-M3 Hybrid Search Weights
BGE_M3_DENSE_WEIGHT=0.3       # Semantic similarity
BGE_M3_SPARSE_WEIGHT=0.5      # Keyword matching (higher for precision)
BGE_M3_COLBERT_WEIGHT=0.2     # Token-level matching

# Performance
BGE_M3_USE_FP16=true          # Faster processing with FP16
BGE_M3_MAX_LENGTH=8192        # Max tokens per document
RERANK_TOP_K=20               # Number of results to rerank
```

### Model Options

**LLM Models (OLLAMA_MODEL):**
- `gemma2:2b` - Fastest, lowest resource usage (recommended for quick responses)
- `llama3.1:8b` - Balanced performance and quality
- `qwen2.5:7b` - Great balance, good multilingual support
- `qwen2.5:14b` - Best quality, higher resource usage

**Embedding Model:**
- BGE-M3 is used automatically (FlagEmbedding), no configuration needed

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

### "FlagEmbedding not found" or import errors
```bash
# Reinstall dependencies
make clean
make setup
```

### Documents not indexed
1. Check Paperless connection:
```bash
curl http://your-paperless-url:8000/api/documents/ \
  -H "Authorization: Token YOUR_TOKEN"
```
2. Verify documents have content (OCR completed)
3. Check logs when running `make index`

### BGE-M3 model download issues
The model (~2.3GB) downloads automatically on first use. To pre-download:
```bash
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

### GPU not being used
Check if Metal Performance Shaders (MPS) is available:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
If false, indexing will fallback to CPU (slower but still works).

### Slow queries
1. Try a smaller LLM model: `OLLAMA_MODEL=gemma2:2b`
2. Reduce chunk size: `CHUNK_SIZE=1000`
3. Adjust weights to favor dense search: `BGE_M3_DENSE_WEIGHT=0.6`

### Out of memory
1. Use a smaller model: `OLLAMA_MODEL=gemma2:2b`
2. Enable FP16: `BGE_M3_USE_FP16=true` (enabled by default)
3. Close other applications

## Example Queries

### Simple Queries (Standard Mode)
- "What invoices do I have from 2024?"
- "Summarize my tax documents"
- "Find all contracts related to insurance"
- "What documents mention warranty?"
- "Show me correspondence from John Doe"

### Analytical Queries (Enhanced Mode)
For queries requiring analysis across **many documents** (e.g., monthly reports, salary slips, invoices):

**Effective queries:**
- "Liste **alle** Gehaltsabrechnungen 2017 mit Lohnsteuer auf"
- "Zeige mir eine **Übersicht** aller Steuerzahlungen pro Monat 2017"
- "Erstelle eine **Tabelle** mit allen Rechnungen und Beträgen aus 2024"
- "**Jeder Monat**: Wie hoch war meine Lohnsteuer 2017?"

**Trigger words** for enhanced retrieval (automatically detected):
- `alle` / `all`
- `gesamt` / `total`
- `übersicht` / `overview`
- `tabelle` / `table`
- `liste` / `list`
- `pro monat` / `per month`
- `jeden monat` / `each month`

When these keywords are detected, the system will:
1. Retrieve **up to 50 chunks** instead of 5-10
2. Instruct the LLM to analyze **each document individually**
3. Format results as **tables or lists**
4. Provide a **summary/total** at the end

**Pro Tip:** For monthly salary data:
```
"Erstelle eine Tabelle: 2017 Gehaltsabrechnungen - 
Monat | Dokument | Lohnsteuer"
```

The chatbot will extract data from each payslip and present it in table format.

## Performance

**Apple Silicon (M-series) with GPU acceleration:**
- **First Run**: BGE-M3 model download (~2.3GB, one-time)
- **Indexing**: ~5-30 docs/minute with GPU (5-10x faster than CPU)
- **Query**: 2-6 seconds average (hybrid search + reranking)
- **Memory**: ~8-12 GB (LLM + BGE-M3 + embeddings)

**Recommended Settings:**
- **LLM**: `gemma2:2b` (fast) or `qwen2.5:7b` (balanced)
- **Chunk size**: 2000 tokens (BGE-M3 handles up to 8192)
- **Retrieval weights**: [0.3, 0.5, 0.2] (dense, sparse, colbert)
- **Use FP16**: Enabled by default for speed

## Advanced Features

### Re-indexing with Existing Vectors

Keep your existing vector database when adding new documents:
```bash
python src/indexer.py --keep-existing
```

### Webhook Integration

Add to Paperless-NGX webhook settings:
```
URL: http://YOUR_IP:8001/webhook/document-added
Event: Document Added
```

### Custom Retrieval Weights

Adjust in `.env` based on your use case:
- **More semantic search**: Increase `BGE_M3_DENSE_WEIGHT`
- **Exact keyword matching**: Increase `BGE_M3_SPARSE_WEIGHT`
- **Token-level precision**: Increase `BGE_M3_COLBERT_WEIGHT`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- [Paperless-NGX](https://github.com/paperless-ngx/paperless-ngx) - Document management system
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) - Multilingual embedding model
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [LangChain](https://python.langchain.com/) - RAG framework

## Quick start

1. Clone and install dependencies:

```bash
git clone <repo>
cd paperless-rag-chatbot
make setup
```

2. Configure:

```bash
cp .env.example .env
# Edit .env and set PAPERLESS_URL and PAPERLESS_TOKEN
```

3. (Optional) download recommended models:

```bash
make setup-models
```

4. Index documents:

```bash
make index
```

5. Start services:

```bash
make serve-all
# web UI: http://localhost:7860
# API docs: http://localhost:8001/docs
```

Where to look next
- `.env.example` — runtime defaults
- `Makefile` — main commands (`setup`, `index`, `serve-all`)
- `src/indexer.py` — indexing logic
- `src/api/server.py` — REST API

Troubleshooting (short)
- Ollama: `ollama serve` and check `http://localhost:11434/api/tags`
- Paperless: verify `PAPERLESS_URL` and `PAPERLESS_TOKEN`
- GPU on macOS: `python -c "import torch; print(torch.backends.mps.is_available())"`

License: MIT

Links
- Paperless‑NGX: https://github.com/paperless-ngx/paperless-ngx
- Ollama: https://ollama.ai/
