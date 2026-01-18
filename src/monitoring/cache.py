"""
Query Cache for RAG System.

Features:
- LRU eviction (keep most recently used)
- TTL expiration (1 hour default)
- Exact match caching
- Fuzzy match for similar queries (85% similarity)
- Thread-safe
- Cache statistics tracking

Usage:
    cache = QueryCache(max_size=1000, ttl_seconds=3600)
    
    # Try to get cached result
    result = cache.get("Wann ist Mavi geboren?")
    if result:
        return result  # Instant response!
    
    # Cache miss - execute query and cache result
    result = execute_query(query)
    cache.set(query, result)
"""

import time
import hashlib
import threading
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import difflib
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    
    query: str
    result: Dict[str, Any]
    timestamp: datetime
    ttl_seconds: int
    hits: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        age = datetime.utcnow() - self.timestamp
        return age.total_seconds() > self.ttl_seconds
    
    def access(self):
        """Mark entry as accessed."""
        self.hits += 1
        self.last_accessed = datetime.utcnow()


class QueryCache:
    """
    LRU cache with TTL for RAG query results.
    
    Features:
    - Exact matching (instant lookup)
    - Fuzzy matching (85% similarity threshold)
    - Automatic expiration
    - Thread-safe operations
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,  # 1 hour
        fuzzy_threshold: float = 0.85  # 85% similarity
    ):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time to live in seconds (default: 1 hour)
            fuzzy_threshold: Similarity threshold for fuzzy matching (0.0-1.0)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.fuzzy_threshold = fuzzy_threshold
        
        # Cache storage (LRU ordered)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'exact_hits': 0,
            'fuzzy_hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
        }
        
        logger.info(f"QueryCache initialized: max_size={max_size}, ttl={ttl_seconds}s, fuzzy_threshold={fuzzy_threshold}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        # Lowercase, strip whitespace, remove extra spaces
        return ' '.join(query.lower().strip().split())
    
    def _compute_hash(self, query: str) -> str:
        """Compute hash for cache key."""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _find_fuzzy_match(self, query: str) -> Optional[str]:
        """
        Find similar query in cache using fuzzy matching.
        
        Args:
            query: Query to match
            
        Returns:
            Cache key of similar query, or None
        """
        normalized_query = self._normalize_query(query)
        query_words = normalized_query.split()
        
        # Check each cached query for similarity
        for cache_key, entry in self._cache.items():
            normalized_cached = self._normalize_query(entry.query)
            cached_words = normalized_cached.split()
            
            # IMPORTANT: If queries contain proper nouns (capitalized words), 
            # require EXACT match on those words to prevent "Luke" matching "Leia"
            original_query_words = query.split()
            original_cached_words = entry.query.split()
            
            has_proper_nouns = any(w[0].isupper() for w in original_query_words if w)
            has_cached_proper_nouns = any(w[0].isupper() for w in original_cached_words if w)
            
            if has_proper_nouns or has_cached_proper_nouns:
                # Extract proper nouns (capitalized words not at start)
                query_nouns = {w.lower() for i, w in enumerate(original_query_words) if i > 0 and w and w[0].isupper()}
                cached_nouns = {w.lower() for i, w in enumerate(original_cached_words) if i > 0 and w and w[0].isupper()}
                
                # If proper nouns don't match, skip fuzzy matching
                if query_nouns != cached_nouns:
                    continue
            
            # Use SequenceMatcher for similarity
            similarity = difflib.SequenceMatcher(
                None,
                normalized_query,
                normalized_cached
            ).ratio()
            
            if similarity >= self.fuzzy_threshold:
                logger.debug(f"Fuzzy match found: '{query}' ~= '{entry.query}' (similarity: {similarity:.2f})")
                return cache_key
        
        return None
    
    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self.stats['expirations'] += 1
            logger.debug(f"Evicted expired entry: {key}")
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if len(self._cache) >= self.max_size:
            # OrderedDict maintains insertion order
            # Move to end on access, so first item is LRU
            lru_key, _ = self._cache.popitem(last=False)
            self.stats['evictions'] += 1
            logger.debug(f"Evicted LRU entry: {lru_key}")
    
    def get(self, query: str, fuzzy: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get cached result for query.
        
        Args:
            query: Query string
            fuzzy: Enable fuzzy matching (default: True)
            
        Returns:
            Cached result dict, or None if not found
        """
        with self._lock:
            self.stats['total_queries'] += 1
            
            # Clean up expired entries periodically
            if self.stats['total_queries'] % 10 == 0:
                self._evict_expired()
            
            # Try exact match first
            cache_key = self._compute_hash(query)
            
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                if entry.is_expired():
                    del self._cache[cache_key]
                    self.stats['expirations'] += 1
                else:
                    # Cache hit!
                    entry.access()
                    # Move to end (most recently used)
                    self._cache.move_to_end(cache_key)
                    self.stats['exact_hits'] += 1
                    
                    logger.info(f"Cache HIT (exact): '{query}' (hits: {entry.hits})")
                    return entry.result
            
            # Try fuzzy match if enabled
            if fuzzy:
                fuzzy_key = self._find_fuzzy_match(query)
                
                if fuzzy_key and fuzzy_key in self._cache:
                    entry = self._cache[fuzzy_key]
                    
                    if entry.is_expired():
                        del self._cache[fuzzy_key]
                        self.stats['expirations'] += 1
                    else:
                        # Fuzzy cache hit!
                        entry.access()
                        self._cache.move_to_end(fuzzy_key)
                        self.stats['fuzzy_hits'] += 1
                        
                        logger.info(f"Cache HIT (fuzzy): '{query}' -> '{entry.query}' (hits: {entry.hits})")
                        return entry.result
            
            # Cache miss
            self.stats['misses'] += 1
            logger.debug(f"Cache MISS: '{query}'")
            return None
    
    def set(self, query: str, result: Dict[str, Any]):
        """
        Cache query result.
        
        Args:
            query: Query string
            result: Result dictionary to cache
        """
        with self._lock:
            cache_key = self._compute_hash(query)
            
            # Evict LRU if at capacity
            self._evict_lru()
            
            # Create new entry
            entry = CacheEntry(
                query=query,
                result=result,
                timestamp=datetime.utcnow(),
                ttl_seconds=self.ttl_seconds
            )
            
            # Add to cache (at end = most recently used)
            self._cache[cache_key] = entry
            
            logger.debug(f"Cached query: '{query}' (cache size: {len(self._cache)})")
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total_hits = self.stats['exact_hits'] + self.stats['fuzzy_hits']
            total_requests = self.stats['total_queries']
            
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'total_queries': total_requests,
                'exact_hits': self.stats['exact_hits'],
                'fuzzy_hits': self.stats['fuzzy_hits'],
                'total_hits': total_hits,
                'misses': self.stats['misses'],
                'hit_rate': round(hit_rate, 2),
                'evictions': self.stats['evictions'],
                'expirations': self.stats['expirations'],
            }
    
    def get_top_queries(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequently cached queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of (query, hit_count) tuples
        """
        with self._lock:
            # Sort by hits descending
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].hits,
                reverse=True
            )
            
            return [(entry.query, entry.hits) for _, entry in sorted_entries[:limit]]


# Global cache instance
_cache_instance: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    """
    Get global query cache instance (singleton).
    
    Returns:
        QueryCache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        from src.config import settings
        
        # Read cache settings from config (with defaults)
        max_size = getattr(settings, 'CACHE_MAX_SIZE', 1000)
        ttl_seconds = getattr(settings, 'CACHE_TTL_SECONDS', 3600)
        fuzzy_threshold = getattr(settings, 'CACHE_FUZZY_THRESHOLD', 0.85)
        
        _cache_instance = QueryCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            fuzzy_threshold=fuzzy_threshold
        )
    
    return _cache_instance
