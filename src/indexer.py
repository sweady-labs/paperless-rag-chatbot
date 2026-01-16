import sys
import os
from src.config import settings

# Add parent directory to path to enable imports
# (keep for backwards compatibility; will remove later)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.paperless_client import PaperlessClient
from src.rag.chunker import DocumentChunker
from src.rag.hybrid_vector_store import HybridVectorStore
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self):
        """Initialize document indexer with BGE-M3 hybrid vector store."""
        self.paperless = PaperlessClient()
        self.chunker = DocumentChunker()
        logger.info("Using BGE-M3 Hybrid Vector Store with GPU acceleration")
        self.vector_store = HybridVectorStore()
    
    def index_all_documents(self, clear_existing: bool = False):
        """Index all documents from Paperless-NGX"""
        if clear_existing:
            logger.info("Clearing existing vector store...")
            self.vector_store.clear_collection()
        
        logger.info("Fetching documents from Paperless-NGX...")
        documents = self.paperless.get_all_documents()
        logger.info(f"Found {len(documents)} documents")
        
        successful = 0
        failed = 0
        
        for doc in tqdm(documents, desc="Indexing documents"):
            try:
                # Skip documents without content
                if not doc.get('content'):
                    logger.warning(f"Skipping document {doc['id']} - no content")
                    continue
                
                # Chunk document
                chunks = self.chunker.chunk_document(doc)
                
                # Add to vector store
                self.vector_store.add_chunks(chunks)
                successful += 1
                
            except Exception as e:
                logger.error(f"Error indexing document {doc['id']}: {e}")
                failed += 1
        
        logger.info(f"Indexing complete! Successfully indexed: {successful}, Failed: {failed}")
        
        # Show collection stats
        info = self.vector_store.get_collection_info()
        logger.info(f"Collection stats: {info['points_count']} chunks indexed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Index Paperless documents with BGE-M3 hybrid search')
    parser.add_argument('--keep-existing', action='store_true',
                       help='Keep existing vectors instead of clearing')
    args = parser.parse_args()
    
    indexer = DocumentIndexer()
    indexer.index_all_documents(clear_existing=not args.keep_existing)
