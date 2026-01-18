"""
Structured logging system for query processing and performance monitoring.

Features:
- JSON-formatted logs for easy parsing
- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Query-specific logging with unique IDs
- Stage-based logging for detailed pipeline visibility
- Automatic log rotation
"""

import logging
import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from logging.handlers import RotatingFileHandler

from ..config import settings


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'query_id'):
            log_data['query_id'] = record.query_id
        if hasattr(record, 'stage'):
            log_data['stage'] = record.stage
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        if hasattr(record, 'details'):
            log_data['details'] = record.details
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
    format_json: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        max_bytes: Max size before rotation
        backup_count: Number of backup files to keep
        format_json: Use JSON formatting
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
    
    # Set formatters
    if format_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Create default loggers
query_logger = setup_logger('query', './logs/query_processing.log', logging.DEBUG)
performance_logger = setup_logger('performance', './logs/performance.log', logging.INFO)
error_logger = setup_logger('error', './logs/errors.log', logging.ERROR)


def get_logger(name: str = 'query') -> logging.Logger:
    """
    Get a logger by name.
    
    Args:
        name: Logger name ('query', 'performance', or 'error')
    
    Returns:
        Logger instance
    """
    loggers = {
        'query': query_logger,
        'performance': performance_logger,
        'error': error_logger,
    }
    return loggers.get(name, query_logger)


class QueryLogger:
    """
    Context manager for logging query processing stages.
    
    Usage:
        with QueryLogger() as qlog:
            qlog.log_stage('rewriting', duration_ms=2.5, details={...})
            qlog.log_stage('search', duration_ms=145, details={...})
    """
    
    def __init__(self, query: str = "", logger: Optional[logging.Logger] = None):
        """
        Initialize query logger.
        
        Args:
            query: User query string
            logger: Custom logger (defaults to query_logger)
        """
        self.query_id = str(uuid.uuid4())[:8]
        self.query = query
        self.logger = logger or query_logger
        self.start_time = time.time()
        self.stages = []
    
    def __enter__(self):
        """Enter context manager."""
        self.log_stage(
            'query_start',
            details={'query': self.query}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        total_duration = (time.time() - self.start_time) * 1000
        
        if exc_type is not None:
            # Log error if exception occurred
            self.logger.error(
                f"Query failed: {exc_val}",
                extra={
                    'query_id': self.query_id,
                    'stage': 'error',
                    'duration_ms': total_duration,
                    'details': {
                        'exception_type': exc_type.__name__,
                        'exception_message': str(exc_val),
                    }
                },
                exc_info=True
            )
        else:
            # Log successful completion
            self.log_stage(
                'query_complete',
                duration_ms=total_duration,
                details={
                    'total_stages': len(self.stages),
                    'stages': self.stages
                }
            )
    
    def log_stage(
        self,
        stage: str,
        duration_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a query processing stage.
        
        Args:
            stage: Stage name (e.g., 'rewriting', 'search', 'llm')
            duration_ms: Stage duration in milliseconds
            details: Additional stage details
        """
        self.stages.append({
            'stage': stage,
            'duration_ms': duration_ms,
            'timestamp': time.time()
        })
        
        extra = {
            'query_id': self.query_id,
            'stage': stage,
        }
        
        if duration_ms is not None:
            extra['duration_ms'] = round(duration_ms, 2)
        
        if details is not None:
            extra['details'] = details
        
        self.logger.info(
            f"[{stage}] {self.query[:50]}...",
            extra=extra
        )
    
    def log_search_results(
        self,
        num_candidates: int,
        num_filtered: int,
        top_score: float,
        avg_score: float,
        vector_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
        duration_ms: Optional[float] = None
    ):
        """
        Log search results with detailed metrics.
        
        Args:
            num_candidates: Number of candidates retrieved
            num_filtered: Number after filtering
            top_score: Best result score
            avg_score: Average score
            vector_score: Top vector search score
            bm25_score: Top BM25 score
            duration_ms: Search duration
        """
        details = {
            'candidates': num_candidates,
            'filtered': num_filtered,
            'top_score': round(top_score, 3),
            'avg_score': round(avg_score, 3),
        }
        
        if vector_score is not None:
            details['vector_score'] = round(vector_score, 3)
        if bm25_score is not None:
            details['bm25_score'] = round(bm25_score, 3)
        
        self.log_stage('search_results', duration_ms=duration_ms, details=details)
    
    def log_llm_generation(
        self,
        tokens_input: int,
        tokens_output: int,
        duration_ms: float,
        tokens_per_sec: float,
        model: str
    ):
        """
        Log LLM generation metrics.
        
        Args:
            tokens_input: Input token count
            tokens_output: Output token count
            duration_ms: Generation duration
            tokens_per_sec: Generation speed
            model: Model name
        """
        details = {
            'model': model,
            'tokens_input': tokens_input,
            'tokens_output': tokens_output,
            'tokens_per_sec': round(tokens_per_sec, 1),
            'total_tokens': tokens_input + tokens_output,
        }
        
        self.log_stage('llm_generation', duration_ms=duration_ms, details=details)
    
    def log_rewrite(
        self,
        original: str,
        rewritten: str,
        patterns_applied: list,
        duration_ms: float
    ):
        """
        Log query rewriting details.
        
        Args:
            original: Original query
            rewritten: Rewritten query
            patterns_applied: List of patterns that matched
            duration_ms: Rewriting duration
        """
        details = {
            'original': original,
            'rewritten': rewritten,
            'patterns_applied': patterns_applied,
            'changed': original.lower() != rewritten.lower(),
        }
        
        self.log_stage('rewriting', duration_ms=duration_ms, details=details)
    
    def log_cache(self, hit: bool, key: Optional[str] = None):
        """
        Log cache hit/miss.
        
        Args:
            hit: Whether cache hit
            key: Cache key (optional)
        """
        details = {
            'hit': hit,
            'status': 'HIT' if hit else 'MISS',
        }
        
        if key:
            details['key'] = key
        
        self.log_stage('cache', details=details)


# Export convenience function
def create_query_logger(query: str) -> QueryLogger:
    """
    Create a query logger for a specific query.
    
    Args:
        query: User query string
    
    Returns:
        QueryLogger instance
    """
    return QueryLogger(query)
