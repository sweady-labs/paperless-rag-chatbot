#!/usr/bin/env python3
"""Fast re-indexing with progress tracking."""

from src.indexer import FastIndexer
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    print("="*80)
    print("FAST RE-INDEX WITH IMPROVED GARBAGE FILTER")
    print("="*80)
    print("\nThis will:")
    print("1. Delete existing index")
    print("2. Re-index all documents with improved garbage detection")
    print("3. Skip chunks with cipher-like text patterns")
    print()
    
    indexer = FastIndexer()
    
    try:
        logger.info("Starting re-index...")
        indexer.index_all_documents()
        logger.info("Re-index complete!")
        
    except KeyboardInterrupt:
        logger.warning("\nRe-index interrupted by user!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Re-index failed: {e}")
        raise

if __name__ == "__main__":
    main()
