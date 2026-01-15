import requests
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class PaperlessClient:
    def __init__(self):
        self.base_url = os.getenv('PAPERLESS_URL')
        self.token = os.getenv('PAPERLESS_TOKEN')
        self.headers = {'Authorization': f'Token {self.token}'}
    
    def get_document_url(self, doc_id: int) -> str:
        """Get the Paperless web UI URL for a document"""
        return f"{self.base_url}/documents/{doc_id}"
    
    def get_all_documents(self, page_size: int = 100) -> List[Dict]:
        """Fetch all documents with pagination"""
        documents = []
        page = 1
        
        while True:
            response = requests.get(
                f"{self.base_url}/api/documents/",
                headers=self.headers,
                params={'page': page, 'page_size': page_size}
            )
            response.raise_for_status()
            data = response.json()
            
            documents.extend(data['results'])
            
            if not data['next']:
                break
            page += 1
        
        return documents
    
    def get_document(self, doc_id: int) -> Dict:
        """Get single document details"""
        response = requests.get(
            f"{self.base_url}/api/documents/{doc_id}/",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def download_document(self, doc_id: int) -> bytes:
        """Download document file"""
        response = requests.get(
            f"{self.base_url}/api/documents/{doc_id}/download/",
            headers=self.headers
        )
        response.raise_for_status()
        return response.content