# Docker Setup for Paperless RAG Chatbot

## Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- Ollama running on your **host machine** (not in Docker)
- Paperless-NGX accessible

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
PAPERLESS_URL=http://localhost:8000
PAPERLESS_TOKEN=your_token_here
OLLAMA_MODEL=gemma2:2b  # Or your preferred model
```

### 3. Start with Docker Compose

**Basic startup (no auto-indexing):**
```bash
docker-compose up -d
```

**With auto-indexing on first run:**
```bash
AUTO_INDEX=true docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f paperless-chatbot
```

### 4. Access the Application

- **Web Interface:** http://localhost:7860
- **API Server:** http://localhost:8001
- **API Docs:** http://localhost:8001/docs

## Configuration Options

Edit `docker-compose.yml` environment section:

```yaml
environment:
  # Auto-index documents on container startup
  - AUTO_INDEX=true  # or false (default)
  
  # Automatically check for new documents every X minutes
  - AUTO_REINDEX_INTERVAL=0  # 0=disabled, e.g., 60 for hourly
  
  # Ollama URL (usually on host machine)
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## Manual Operations

### Index/Re-index Documents
```bash
docker exec paperless-rag-chatbot python3 -m src.indexer
```

### Update URLs in Existing Database
```bash
docker exec paperless-rag-chatbot python3 update_urls.py
```

### Check Health
```bash
curl http://localhost:8001/health
```

### View Container Logs
```bash
docker-compose logs -f
```

## Troubleshooting

### Ollama Not Reachable
```
❌ ERROR: Ollama is not reachable
```

**Solution:**
1. Make sure Ollama is running on host: `ollama serve`
2. Check if model is available: `ollama list`
3. Pull model if needed: `ollama pull gemma2:2b`

### Model Not Found
```
⚠️ WARNING: Model gemma2:2b not found
```

**Solution:**
```bash
# On your host machine:
ollama pull gemma2:2b
```

### Paperless Connection Failed
```
❌ ERROR: Cannot connect to Paperless
```

**Solution:**
1. Check PAPERLESS_URL in `.env`
2. Verify PAPERLESS_TOKEN is correct
3. Ensure Paperless is accessible from Docker

### No Vector Database
```
⚠️ No vector database found - indexing required
```

**Solution:**
```bash
# Option 1: Enable auto-indexing
AUTO_INDEX=true docker-compose up -d

# Option 2: Run indexing manually
docker exec paperless-rag-chatbot python3 -m src.indexer
```

## Updating the Application

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

## Stopping the Application

```bash
# Stop containers
docker-compose down

# Stop and remove volumes (WARNING: deletes vector database)
docker-compose down -v
```

## Data Persistence

The vector database is stored in `./data/` and persists between container restarts. To backup:

```bash
tar -czf chatbot-backup-$(date +%Y%m%d).tar.gz data/
```

## Performance Tips

1. **Use a lightweight model** for faster responses:
   - `gemma2:2b` (recommended, 1.6GB)
   - `phi3:3.8b` (good quality, 2.2GB)

2. **Adjust resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

3. **Disable auto-indexing** if you don't add documents frequently

## Production Deployment

For production use:

1. Use external Qdrant server instead of local storage
2. Set up proper reverse proxy (nginx/traefik)
3. Enable SSL/TLS
4. Configure log rotation
5. Set up monitoring and alerting

Example with external Qdrant:
```yaml
environment:
  - QDRANT_URL=http://qdrant-server:6333
  - VECTOR_DB_PATH=  # Leave empty to use external Qdrant
```
