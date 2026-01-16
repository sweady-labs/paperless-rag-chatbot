"""
Document chunker optimized for fast RAG.

Uses smaller chunks for faster LLM processing.
"""

from typing import List, Dict, Optional
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from src.config import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Document chunker for fast RAG pipeline.
    
    Uses smaller chunks (512 tokens) for:
    - Faster LLM processing (less context)
    - More precise retrieval
    - Better cache efficiency
    """
    
    def __init__(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None, 
        max_chunk_size: Optional[int] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Tokens per chunk (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            max_chunk_size: Maximum allowed chunk size (default from settings)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.max_chunk_size = max_chunk_size or settings.MAX_CHUNK_SIZE
        
        # Use tiktoken for accurate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=lambda text: len(self.encoding.encode(text)),
            separators=["\n\n", "\n", ". ", " "]
        )
        
        logger.info(
            f"Chunker: size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"max={self.max_chunk_size}"
        )
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a Paperless document into smaller pieces with metadata.
        
        Args:
            document: Paperless document dict with 'content', 'title', etc.
        
        Returns:
            List of chunks with metadata
        """
        content = document.get('content', '')
        
        if not content:
            return []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Add metadata to each chunk
        chunked_docs = []
        skipped_chunks = 0
        
        # Generate Paperless document URL
        paperless_url = settings.PAPERLESS_URL
        doc_url = f"{paperless_url}/documents/{document['id']}"
        
        for i, chunk in enumerate(chunks):
            # Check chunk size
            chunk_tokens = len(self.encoding.encode(chunk))
            
            if chunk_tokens > self.max_chunk_size:
                skipped_chunks += 1
                continue
            
            chunked_docs.append({
                'text': chunk,
                'metadata': {
                    'document_id': document['id'],
                    'title': document.get('title', 'Untitled'),
                    'created': document.get('created'),
                    'modified': document.get('modified'),
                    'correspondent': document.get('correspondent_name'),
                    'tags': document.get('tags', []),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source': 'paperless-ngx',
                    'url': doc_url
                }
            })
        
        if skipped_chunks > 0:
            logger.warning(
                f"Skipped {skipped_chunks} oversized chunks from document {document['id']}"
            )
        
        return chunked_docs
