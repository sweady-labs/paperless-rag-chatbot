from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import os
from src.config import settings

class DocumentChunker:
    """
    Document chunker optimized for BGE-M3 model.
    
    BGE-M3 supports up to 8192 tokens, but for optimal performance:
    - Use 1000-2000 tokens per chunk for general documents
    - Use larger chunks (up to 4000) for long documents with context
    - Increase overlap to preserve context across chunks
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, max_chunk_size: int = 8000):
        # Load from environment or use BGE-M3 optimized defaults
        if chunk_size is None:
            chunk_size = settings.CHUNK_SIZE  # Optimal for BGE-M3
        if chunk_overlap is None:
            chunk_overlap = settings.CHUNK_OVERLAP  # 20% overlap recommended
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size  # BGE-M3 max is 8192
        
        # Use tiktoken for accurate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.encoding.encode(text)),
            separators=["\n\n", "\n", ". ", " "]
        )
        
        print(f"📝 Chunker initialized: size={chunk_size}, overlap={chunk_overlap}, max={max_chunk_size}")
    
    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Chunk a Paperless document into smaller pieces with metadata.
        Optimized for BGE-M3's 8K token context window.
        
        Args:
            document: Paperless document dict with 'content', 'title', etc.
        
        Returns:
            List of chunks with metadata
        """
        content = document.get('content', '')
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Add metadata to each chunk, filtering out oversized chunks
        chunked_docs = []
        skipped_chunks = 0
        
        for i, chunk in enumerate(chunks):
            # Check chunk size to avoid embedding API errors
            chunk_tokens = len(self.encoding.encode(chunk))
            
            if chunk_tokens > self.max_chunk_size:
                # BGE-M3 supports up to 8192, but we cap at 8000 for safety
                skipped_chunks += 1
                continue  # Skip chunks that are too large
            
            # Generate Paperless document URL
            paperless_url = settings.PAPERLESS_URL
            doc_url = f"{paperless_url}/documents/{document['id']}"
            
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
                    'url': doc_url  # Direct link to document in Paperless
                }
            })
        
        if skipped_chunks > 0:
            import logging
            logging.warning(f"Skipped {skipped_chunks} oversized chunks from document {document['id']}")
        
        return chunked_docs
