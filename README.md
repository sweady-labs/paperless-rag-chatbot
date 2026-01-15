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

## Quick Start (Docker - Recommended)

**Easiest way to run:**

```bash
# 1. Configure your settings
cp .env.example .env
# Edit .env with your Paperless URL and token

# 2. Make sure Ollama is running on your host
ollama serve

# 3. Start everything with Docker
docker-compose up -d

# 4. Access the Web Interface
# Open http://localhost:7860
```

See [README-DOCKER.md](README-DOCKER.md) for detailed Docker documentation.

## Alternative: Manual Installation

### Prerequisites

- Python 3.11+
- Ollama installed on Mac
- Paperless-NGX running (http://192.168.178.111:8000)
- Paperless API token

## Quick Start (Manual)

### 1. Install Dependencies

```bash
cd ~/paperless-rag-chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama Model

```bash
# Install the lightweight model (recommended)
ollama pull gemma2:2b

# Or use a more powerful model if you need better quality
# ollama pull llama3.1:8b

# Note: BGE-M3 embeddings are downloaded automatically via HuggingFace
```

### 3. Configure Environment

The `.env` file is already configured. Verify it contains:
- PAPERLESS_URL=http://192.168.178.111:8000
- PAPERLESS_TOKEN=your_token
- OLLAMA_MODEL=gemma2:2b  # For fast responses

### 4. Start the Chatbot

```bash
# Use the all-in-one startup script
./start.sh

# Keep existing vectors instead of clearing
python src/indexer.py --keep-existing
```

This will:
- Fetch all documents from Paperless-NGX
- Chunk them into optimal sizes for BGE-M3 (up to 8K tokens)
- Generate dense, sparse, and ColBERT embeddings
- Store in Qdrant vector database with Named Vectors support

### 5. Start the API Server

```bash
python src/api/server.py
```

Server will start at: http://localhost:8001

### 6. Use the Chatbot

**Option A: CLI Interface**


Select from menu:
1) Index documents (first time or update)
2) Start API server
3) Start Web Interface
4) Start CLI Chat
5) Debug: Test query
```

This will:
- Check if Ollama is running
- Verify required models
- Start your selected interface

### Alternative: Manual Steps

**Start API Server**
```bash
python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8001
```

**Start Web Interface**
```bash
python src/web_interface.py
```
Then open: http://localhost:7860

**Start CLI Chat**
```bash
python src/cli_chat.py
```

## API Endpoints

- `GET /` - API information
- `POST /query` - Query documents
- `GET /health` - Health check & stats
- `GET /stats` - Indexing statistics
- `GET /docs` - Interactive API documentation (Swagger)
- `POST /reindex` - Re-index all documents

## Project Structure

```
paperless-rag-chatbot/
├── .env                    # Configuration
├── requirements.txt        # Python dependencies
├── start.sh               # All-in-one startup script
├── src/
│   ├── api/
│   │   ├── paperless_client.py  # Paperless API integration
│   │   └── server.py            # FastAPI server
│   ├── rag/
│   │   ├── chunker.py           # Document chunking
│   │   ├── vector_store.py      # Dense-only Qdrant store
│   │   ├── hybrid_vector_store.py  # BGE-M3 hybrid store (used by default)
│   │   ├── query_engine.py      # RAG query logic
│   │   └── reranker.py          # Re-ranking for accuracy
│   ├── indexer.py          # Document indexing script
│   ├── cli_chat.py         # CLI interface
│   └── web_interface.py    # Gradio web UI
└── data/
    └── vector_db/          # Qdrant storage
```

## Configuration

Edit `.env` to customize:

```bash
# Model Selection
OLLAMA_MODEL=gemma2:2b             # Lightweight and fast
# Alternative models: llama3.1:8b, qwen2.5:7b, phi3:3.8b

OLLAMA_EMBEDDING_MODEL=bge-m3      # BGE-M3 embeddings (auto-downloaded)

# Chunking (optimized for BGE-M3's 8K token window)
CHUNK_SIZE=800                     # Tokens per chunk
CHUNK_OVERLAP=150                  # Overlap between chunks

# BGE-M3 Retrieval Weights (dense, sparse, colbert)
# Default: [0.4, 0.2, 0.4] as recommended by BGE-M3 authors
BGE_M3_WEIGHTS=0.4,0.2,0.4

# Vector DB
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=paperless_documents
```

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama service
ollama serve
```

### "FlagEmbedding not found" or import errors
```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### "Import errors" when running scripts
Make sure you're in the virtual environment:
```bash
source venv/bin/activate
```

### Slow or poor quality answers
1. Try using a larger model: `OLLAMA_MODEL=qwen2.5:14b`
2. Adjust BGE-M3 weights in queries (increase dense weight for semantic, sparse for exact matches)
3. Increase chunk sizes for longer context: `CHUNK_SIZE=2000`
4. Use re-ranking (enabled by default)

### Documents not indexed
1. Check Paperless connection: `curl http://192.168.178.111:8000/api/documents/ -H "Authorization: Token YOUR_TOKEN"`
2. Verify documents have content (OCR completed)
3. Check logs when running `python src/indexer.py`

### BGE-M3 model download issues
The model (~2.3GB) downloads automatically on first use. If you have network issues:
```bash
# Pre-download the model
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

## Example Queries

- "What invoices do I have from 2024?"
- "Summarize my tax documents"
- "Find all contracts related to insurance"
- "What documents mention warranty?"
- "Show me correspondence from John Doe"

## Performance

**Mac M5 Pro with BGE-M3:**
- **First Run**: BGE-M3 model download (~2.3GB, one-time)
- **Indexing**: ~5-30 docs/minute (BGE-M3 is more thorough)
- **Query**: 2-6 seconds average (hybrid search + reranking)
- **Memory**: ~8-12 GB (LLM + BGE-M3 + embeddings)

**Recommended Settings:**
- **LLM**: qwen2.5:7b (balanced) or qwen2.5:14b (best quality)
- **Chunk size**: 1000-2000 tokens (BGE-M3 handles up to 8192)
- **Retrieval weights**: [0.4, 0.2, 0.4] (dense, sparse, colbert)
- **Use FP16**: Enabled by default for speed

**Speed vs Accuracy Trade-offs:**
- Fastest: Dense-only (`--no-hybrid`) ~2-3s/query
- Balanced: Hybrid + fast rerank ~4-5s/query (default)
- Best: Hybrid + LLM rerank ~6-10s/query

## Advanced Features

### Webhook Integration

Add to Paperless-NGX webhook:
```
URL: http://YOUR_MAC_IP:8001/webhook/document-added
Event: Document Added
```

### Docker Deployment

Coming soon - Docker Compose setup for production.

## License

MIT

## Support

- Paperless-NGX Docs: https://docs.paperless-ngx.com/
- Ollama: https://ollama.ai/
- LangChain: https://python.langchain.com/
