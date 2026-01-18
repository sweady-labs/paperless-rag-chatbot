# Test Suite

This folder contains all test scripts for the Paperless RAG Chatbot.

## Test Files

### Core Functionality Tests

- **`comprehensive_test.py`** - Full system test with 24 comprehensive queries
  - Tests all major features end-to-end
  - Validates query success rates
  - Run: `python tests/comprehensive_test.py`

- **`test_cache.py`** - Query cache functionality tests
  - Tests cache hit/miss behavior
  - Validates fuzzy matching
  - Tests TTL expiration
  - Run: `python tests/test_cache.py`

- **`test_synonyms.py`** - Synonym expansion tests
  - Tests German synonym expansion
  - Validates expansion detection logic
  - Tests real-world scenarios
  - Run: `python tests/test_synonyms.py`

### Monitoring & Performance Tests

- **`test_monitoring.py`** - Monitoring system tests
  - Tests JSON logging
  - Validates metrics collection
  - Tests SQLite storage
  - Run: `python tests/test_monitoring.py`

- **`test_monitored_engine.py`** - Query engine with monitoring
  - Tests query engine with full monitoring
  - Validates metric recording
  - Run: `python tests/test_monitored_engine.py`

- **`test_dashboard.py`** - Dashboard functionality tests
  - Tests dashboard API endpoints
  - Validates metric visualization
  - Run: `python tests/test_dashboard.py`

### Streaming Tests

- **`test_streaming.py`** - LLM streaming tests
  - Tests streaming API endpoint
  - Validates NDJSON format
  - Tests token-by-token delivery
  - Run: `python tests/test_streaming.py`

## Running Tests

### Run Individual Tests
```bash
# From repository root
python tests/test_cache.py
python tests/test_synonyms.py
python tests/comprehensive_test.py
```

### Run All Tests
```bash
# Simple approach
for test in tests/test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

### Prerequisites

1. **Virtual environment activated**
   ```bash
   source .venv/bin/activate
   ```

2. **Services running** (for integration tests)
   ```bash
   make serve-all
   ```

3. **Documents indexed**
   ```bash
   make index
   ```

## Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| **Unit Tests** | `test_cache.py`, `test_synonyms.py` | Test individual components |
| **Integration Tests** | `test_monitored_engine.py`, `test_streaming.py` | Test component interactions |
| **System Tests** | `comprehensive_test.py` | Test entire system end-to-end |
| **UI Tests** | `test_dashboard.py` | Test web interface components |

## Expected Results

All tests should pass with:
- ✅ Cache hit/miss working correctly
- ✅ Synonym expansion for appropriate queries
- ✅ Monitoring metrics being recorded
- ✅ Streaming delivering tokens in real-time
- ✅ Comprehensive test: 100% success rate (24/24 queries)

## Notes

- Tests use the same `.env` configuration as the main application
- Some tests require the API server to be running (`make serve-api`)
- Integration tests may take several minutes to complete
- Check logs in `logs/` folder if tests fail
