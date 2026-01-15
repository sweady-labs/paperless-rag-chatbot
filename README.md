# Paperless RAG Chatbot

AI-powered chatbot for querying documents in Paperless-NGX using RAG (Retrieval-Augmented Generation) with **BGE-M3 hybrid search**.

## Features

- **BGE-M3 Hybrid Search**: Multi-functionality retrieval (dense + sparse + ColBERT)
- **Multilingual Support**: Works with 100+ languages via BGE-M3
- **Long Document Support**: Up to 8192 tokens per chunk
- **Re-ranking**: Improved accuracy with fast keyword-based reranking
- **Local LLM**: Uses Ollama (gemma2:2b for fast responses)
- **Vector Database**: Qdrant for efficient similarity search
- **FastAPI Backend**: RESTful API for integration
- **Multiple Interfaces**: CLI and Web UI (Gradio)
- **Optimized Performance**: Fast queries with low CPU usage
- **Docker Support**: Easy deployment with Docker Compose
- **Clickable Links**: Direct links to documents in Paperless

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Paperless-NGX instance with API access
- Paperless API token ([create one in Paperless settings](https://docs.paperless-ngx.com/api/#authorization))

## Quick Start (Docker - Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/sweady-labs/paperless-rag-chatbot.git
cd paperless-rag-chatbot

# 2. Configure your settings
cp .env.example .env
nano .env  # Edit with your Paperless URL and API token

# 3. Make sure Ollama is running on your host
ollama serve

# 4. Pull the required Ollama model
ollama pull gemma2:2b

# 5. Start everything with Docker
docker-compose up -d

# 6. Access the Web Interface
# Open http://localhost:7860
```

See [README-DOCKER.md](README-DOCKER.md) for detailed Docker documentation.

## Alternative: Manual Installation

### 1. Clone and Setup

```bash
git clone https://github.com/sweady-labs/paperless-rag-chatbot.git
cd paperless-rag-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
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

### 3. Install Ollama Models

```bash
# Install the lightweight model (recommended)
ollama pull gemma2:2b

# Or use a more powerful model for better quality
# ollama pull llama3.1:8b
# ollama pull qwen2.5:7b

# Note: BGE-M3 embeddings are downloaded automatically on first use
```

### 4. Run the Application

**Easy start (recommended):**
```bash
./start.sh
```

Select from menu:
1. Index documents (first time or update)
2. Start API server
3. Start Web Interface
4. Start CLI Chat
5. Debug: Test query

**Or run components manually:**

```bash
# Index documents first
python src/indexer.py

# Start API Server
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001

# Start Web Interface (in another terminal)
python src/web_interface.py
# Then open: http://localhost:7860

# Or start CLI Chat
python src/cli_chat.py
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
├── start.sh               # All-in-one startup script
├── docker-compose.yml     # Docker deployment
├── Dockerfile             # Docker image
├── src/
│   ├── api/
│   │   ├── paperless_client.py  # Paperless API integration
│   │   └── server.py            # FastAPI server
│   ├── rag/
│   │   ├── chunker.py           # Document chunking
│   │   ├── vector_store.py      # Dense-only Qdrant store
│   │   ├── hybrid_vector_store.py  # BGE-M3 hybrid store (default)
│   │   ├── query_engine.py      # RAG query logic
│   │   └── reranker.py          # Re-ranking for accuracy
│   ├── indexer.py          # Document indexing script
│   ├── cli_chat.py         # CLI interface
│   └── web_interface.py    # Gradio web UI
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
OLLAMA_EMBEDDING_MODEL=bge-m3 # BGE-M3 embeddings

# Chunking (optimized for BGE-M3's 8K token window)
CHUNK_SIZE=800                # Tokens per chunk
CHUNK_OVERLAP=150             # Overlap between chunks

# BGE-M3 Hybrid Search Weights
BGE_M3_DENSE_WEIGHT=0.4       # Semantic similarity
BGE_M3_SPARSE_WEIGHT=0.4      # Keyword matching
BGE_M3_COLBERT_WEIGHT=0.2     # Token-level matching

# Performance
BGE_M3_USE_FP16=true          # Faster processing
BGE_M3_MAX_LENGTH=8192        # Max tokens per document
RERANK_TOP_K=10               # Number of results to rerank
```

### Model Options

**LLM Models (OLLAMA_MODEL):**
- `gemma2:2b` - Fastest, lowest resource usage (recommended for quick responses)
- `llama3.1:8b` - Balanced performance and quality
- `qwen2.5:7b` - Great balance, good multilingual support
- `qwen2.5:14b` - Best quality, higher resource usage

**Embedding Model:**
- `bge-m3` - Used automatically, no need to change

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
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### Documents not indexed
1. Check Paperless connection:
```bash
curl http://your-paperless-url:8000/api/documents/ \
  -H "Authorization: Token YOUR_TOKEN"
```
2. Verify documents have content (OCR completed)
3. Check logs when running `python src/indexer.py`

### BGE-M3 model download issues
The model (~2.3GB) downloads automatically on first use. To pre-download:
```bash
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

### Slow queries
1. Try a smaller LLM model: `OLLAMA_MODEL=gemma2:2b`
2. Reduce chunk size: `CHUNK_SIZE=600`
3. Adjust weights to favor dense search: `BGE_M3_DENSE_WEIGHT=0.6`

### Out of memory
1. Use a smaller model: `OLLAMA_MODEL=gemma2:2b`
2. Enable FP16: `BGE_M3_USE_FP16=true`
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

**✅ Effective queries:**
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

**💡 Pro Tip:** For monthly salary data:
```
"Erstelle eine Tabelle: 2017 Gehaltsabrechnungen - 
Monat | Dokument | Lohnsteuer"
```

The chatbot will then extract data from each payslip and present it in table format.

## Performance

**Mac M5 Pro with BGE-M3:**
- **First Run**: BGE-M3 model download (~2.3GB, one-time)
- **Indexing**: ~5-30 docs/minute (depends on document size)
- **Query**: 2-6 seconds average (hybrid search + reranking)
- **Memory**: ~8-12 GB (LLM + BGE-M3 + embeddings)

**Recommended Settings:**
- **LLM**: `gemma2:2b` (fast) or `qwen2.5:7b` (balanced)
- **Chunk size**: 800-2000 tokens (BGE-M3 handles up to 8192)
- **Retrieval weights**: [0.4, 0.4, 0.2] (dense, sparse, colbert)
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

## Support

- Paperless-NGX Docs: https://docs.paperless-ngx.com/
- Ollama: https://ollama.ai/
- BGE-M3: https://github.com/FlagOpen/FlagEmbedding
