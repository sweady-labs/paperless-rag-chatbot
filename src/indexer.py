"""
Fast Document Indexer using Ollama embeddings.

Optimized for Apple Silicon with mxbai-embed-large.
Uses batch indexing to avoid SQLite locking issues.
"""

import sys
import os
import logging
import time

from tqdm import tqdm

from src.config import settings
from src.api.paperless_client import PaperlessClient
from src.rag.chunker import DocumentChunker
from src.rag.fast_vector_store import FastVectorStore, create_point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastIndexer:
    """
    Fast document indexer using Ollama embeddings.
    
    Uses batch indexing: collects all embeddings first, then writes to 
    Qdrant in a single batch to avoid SQLite locking issues.
    """
    
    def __init__(self):
        """Initialize the fast indexer."""
        self.paperless = PaperlessClient()
        self.chunker = DocumentChunker()
        self.vector_store = FastVectorStore()
        
        logger.info("FastIndexer initialized")
        logger.info(f"  - Embedding model: {settings.OLLAMA_EMBEDDING_MODEL}")
        logger.info(f"  - Chunk size: {settings.CHUNK_SIZE}")
        logger.info(f"  - Collection: {settings.COLLECTION_NAME}")
    
    def index_all_documents(self, clear_existing: bool = True):
        """
        Index all documents from Paperless-NGX.
        
        Uses batch indexing: embeds all documents first, then writes once.
        
        Args:
            clear_existing: Clear existing index before re-indexing
        """
        if clear_existing:
            logger.info("Clearing existing vector store...")
            self.vector_store.clear_collection()
        
        logger.info("Fetching documents from Paperless-NGX...")
        documents = self.paperless.get_all_documents()
        logger.info(f"Found {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found in Paperless-NGX")
            return
        
        # PHASE 1: Collect all embeddings in memory
        logger.info("=" * 50)
        logger.info("PHASE 1: Embedding all documents...")
        logger.info("=" * 50)
        
        all_points = []
        successful = 0
        failed = 0
        skipped_chunks = 0
        
        for doc in tqdm(documents, desc="Embedding"):
            try:
                # Skip documents without content
                if not doc.get('content'):
                    logger.debug(f"Skipping document {doc['id']} - no content")
                    continue
                
                # Chunk document
                chunks = self.chunker.chunk_document(doc)
                
                if not chunks:
                    logger.debug(f"No chunks for document {doc['id']}")
                    continue
                
                # Embed each chunk
                doc_points = []
                for chunk in chunks:
                    text = chunk['text']
                    metadata = chunk['metadata']
                    
                    # Embed single text
                    embedding = self.vector_store.embed_chunk(text)
                    if embedding is None:
                        skipped_chunks += 1
                        continue
                    
                    # Create point
                    point = create_point(embedding, text, metadata)
                    doc_points.append(point)
                
                if doc_points:
                    all_points.extend(doc_points)
                    successful += 1
                else:
                    failed += 1
                
            except Exception as e:
                logger.error(f"Error embedding document {doc['id']}: {e}")
                failed += 1
        
        logger.info(f"Embedding complete: {len(all_points)} points from {successful} documents")
        if skipped_chunks > 0:
            logger.warning(f"Skipped {skipped_chunks} chunks due to embedding errors")
        
        if not all_points:
            logger.error("No points to index!")
            return
        
        # PHASE 2: Write all points to Qdrant in a single batch
        logger.info("=" * 50)
        logger.info("PHASE 2: Writing to vector database...")
        logger.info("=" * 50)
        
        try:
            # Write in batches of 1000 to avoid memory issues
            batch_size = 1000
            for i in range(0, len(all_points), batch_size):
                batch = all_points[i:i+batch_size]
                self.vector_store.add_points_batch(batch)
                logger.info(f"Written {min(i+batch_size, len(all_points))}/{len(all_points)} points")
        except Exception as e:
            logger.error(f"Error writing to vector database: {e}")
            raise
        
        # Summary
        logger.info("=" * 50)
        logger.info("Indexing Complete!")
        logger.info(f"  - Documents indexed: {successful}")
        logger.info(f"  - Documents failed: {failed}")
        logger.info(f"  - Total points: {len(all_points)}")
        
        # Show collection stats
        info = self.vector_store.get_collection_info()
        logger.info(f"  - Collection: {info['name']}")
        logger.info(f"  - Points in DB: {info['points_count']}")
    
    def index_single_document(self, document_id: int):
        """
        Index a single document by ID.
        
        Args:
            document_id: Paperless document ID
        """
        logger.info(f"Indexing document {document_id}...")
        
        try:
            doc = self.paperless.get_document(document_id)
            
            if not doc.get('content'):
                logger.warning(f"Document {document_id} has no content")
                return False
            
            chunks = self.chunker.chunk_document(doc)
            self.vector_store.add_chunks(chunks)
            
            logger.info(f"Indexed document {document_id}: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {e}")
            return False


# Keep backwards compatibility alias
DocumentIndexer = FastIndexer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Index Paperless documents with fast Ollama embeddings'
    )
    parser.add_argument(
        '--keep-existing', 
        action='store_true',
        help='Keep existing vectors instead of clearing'
    )
    parser.add_argument(
        '--document-id',
        type=int,
        help='Index a single document by ID'
    )
    args = parser.parse_args()
    
    indexer = FastIndexer()
    
    if args.document_id:
        indexer.index_single_document(args.document_id)
    else:
        indexer.index_all_documents(clear_existing=not args.keep_existing)
