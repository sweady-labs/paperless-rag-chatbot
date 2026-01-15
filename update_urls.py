#!/usr/bin/env python3
"""
Quick script to add URLs to existing indexed documents without re-indexing.
"""
import sys
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

def update_document_urls():
    """Add URL metadata to existing points in Qdrant"""
    
    # Connect to Qdrant
    db_path = os.getenv('VECTOR_DB_PATH', './data/vector_db')
    collection_name = os.getenv('COLLECTION_NAME', 'paperless_documents')
    paperless_url = os.getenv('PAPERLESS_URL', 'http://localhost:8000')
    
    print(f"📦 Connecting to Qdrant at {db_path}...")
    client = QdrantClient(path=db_path)
    
    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Found collection: {collection_name}")
        print(f"   Points: {collection_info.points_count}")
    except Exception as e:
        print(f"❌ Collection not found: {e}")
        return
    
    # Get all points (in batches)
    print("\n🔄 Updating URLs in metadata...")
    
    # Scroll through all points
    offset = None
    updated_count = 0
    batch_size = 100
    
    while True:
        # Get batch of points
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False  # Don't need vectors, saves memory
        )
        
        if not points:
            break
        
        # Update each point
        for point in points:
            if 'document_id' in point.payload:
                doc_id = point.payload['document_id']
                doc_url = f"{paperless_url}/documents/{doc_id}"
                
                # Update payload
                client.set_payload(
                    collection_name=collection_name,
                    payload={'url': doc_url},
                    points=[point.id]
                )
                updated_count += 1
                
                if updated_count % 100 == 0:
                    print(f"   Updated {updated_count} points...")
        
        # Check if we're done
        if next_offset is None:
            break
        offset = next_offset
    
    print(f"\n✅ Updated {updated_count} points with URLs")
    print(f"   All documents now have clickable links!")

if __name__ == "__main__":
    try:
        update_document_urls()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
