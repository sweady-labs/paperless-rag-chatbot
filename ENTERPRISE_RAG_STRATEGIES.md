# Enterprise RAG-Strategien (NotebookLM, OpenAI, etc.)

> Dokumentation moderner RAG-Ansätze von Google, Microsoft, Anthropic und wie wir sie adaptieren können


CHUNK_SIZE=2000              # Statt 1500
CHUNK_OVERLAP=300            # Statt 200
RERANK_TOP_K=20              # Statt 15

## 📋 Übersicht

Diese Strategien werden von führenden Tech-Firmen in ihren RAG-Systemen verwendet:
- Google NotebookLM
- OpenAI Assistants API
- Anthropic Claude + Retrieval
- Microsoft Copilot / Azure OpenAI
- Voyage AI, Cohere

## 🏢 Enterprise-Ansätze im Detail

### 1. Google NotebookLM

**Chunking-Strategie:**
```python
# Adaptive, Document-Aware Chunking
class NotebookLMChunker:
    def chunk(self, document):
        # Keine feste Chunk-Size!
        # Stattdessen: Semantic Boundaries
        
        # 1. Erkenne Dokument-Struktur
        structure = self.detect_structure(document)
        # → PDF: Sections, Headers, Paragraphs
        # → Markdown: # Headers, ## Sub-headers
        # → Text: Absätze, Sätze
        
        # 2. Split an logischen Grenzen
        chunks = self.split_at_boundaries(document, structure)
        
        # 3. Hierarchical Indexing
        return {
            "parent": full_document_embedding,
            "children": chunk_embeddings,
            "metadata": extracted_entities
        }
```

**Hierarchical Retrieval:**
```
Query: "Lohnsteuer Oktober 2017"

Step 1: Search in Child Chunks (präzise)
  → Findet: Chunk "Gehaltsbestandteile Oktober 2017"
  
Step 2: Retrieve Parent Chunk (Kontext)
  → Holt: Vollständige Gehaltsabrechnung
  
Step 3: Multi-Document Linking
  → Findet: Ähnliche Gehaltsabrechnungen (Nov, Dez 2017)
  
Step 4: Aggregation
  → LLM erstellt Tabelle über alle Monate
```

**Key Features:**
- ✅ Versteht Dokumentstruktur (PDF Headers, Sections)
- ✅ Parent-Child Beziehungen
- ✅ Cross-Document Linking
- ✅ Automatic Summarization per Dokument
- ✅ Question Suggestions basierend auf Inhalt

---

### 2. OpenAI Assistants API

**Chunking-Strategie:**
```python
# Very Small Chunks + Function Calling
class OpenAIChunker:
    chunk_size = 512      # Sehr klein!
    overlap = 50
    
    # Code-aware splitting
    preserves = [
        "code_blocks",
        "tables",
        "lists"
    ]
```

**Workflow:**
```
1. User Query: "Alle Gehaltsabrechnungen 2017"

2. Retrieval: Hole relevante Chunks
   → 50-100 kleine Chunks (je 512 tokens)

3. Function Calling:
   {
     "name": "extract_salary_data",
     "parameters": {
       "year": 2017,
       "data_type": "tax"
     }
   }

4. Structured Extraction pro Dokument

5. Aggregation & Tabelle
```

**Vorteile:**
- Sehr präzise Retrieval
- Kombiniert mit strukturierter Daten-Extraktion
- Function Calling für komplexe Queries

**Nachteile:**
- Viele API-Calls = teuer
- Kleine Chunks = mehr Fragmentierung
- Braucht Cloud-Service

---

### 3. Anthropic Claude + Extended Context

**Ansatz:**
```python
# Large Context Window = Weniger Chunking
class ClaudeRAG:
    context_window = 200_000  # 200K tokens!
    
    def query(self, question):
        # 1. Retrieval: Hole relevante DOKUMENTE (nicht Chunks!)
        docs = self.retrieve_documents(question, top_k=50)
        
        # 2. Load FULL documents into context
        full_context = "\n\n---\n\n".join([d.content for d in docs])
        
        # 3. Single LLM call mit ALLEN Dokumenten
        answer = self.llm.invoke(
            prompt=f"Context:\n{full_context}\n\nQuestion: {question}"
        )
        
        # 4. Prompt Caching: Dokumente werden gecacht ($$$ sparen)
        return answer
```

**Beispiel:**
```
Query: "Alle Rechnungen Oktober 2024"

→ Retrieval findet 30 Rechnungs-PDFs
→ Lädt ALLE 30 vollständigen PDFs in Claude (200K window!)
→ Claude liest ALLES auf einmal
→ Erstellt Tabelle direkt aus vollem Kontext

Kein Chunking-Problem!
```

**Vorteile:**
- ✅ Kein Chunk-Fragmentierungs-Problem
- ✅ LLM sieht vollständigen Kontext
- ✅ Prompt Caching spart Kosten

**Nachteile:**
- API-Kosten (teuer bei vielen Docs)
- Nicht lokal möglich (Ollama hat nur ~8K-32K window)

---

### 4. Microsoft Copilot / Semantic Kernel

**Multi-Strategy Chunking:**
```python
from semantic_kernel import chunking

class SemanticKernelChunker:
    strategies = {
        "fixed": FixedSizeChunker(size=1000),
        "sentence": SentenceChunker(),
        "paragraph": ParagraphChunker(),
        "markdown": MarkdownChunker(),
        "sliding": SlidingWindowChunker(window=1500, stride=750),
        "hierarchical": HierarchicalChunker()
    }
    
    def chunk(self, document):
        # Wähle Strategie basierend auf Dokumenttyp
        doc_type = self.detect_type(document)
        strategy = self.select_strategy(doc_type)
        return strategy.chunk(document)
```

**Document Type Detection:**
```python
def detect_type(self, doc):
    # Basierend auf:
    # - Dateiname
    # - Metadata (Paperless Tags!)
    # - Content-Struktur
    # - Length
    
    if "rechnung" in doc.title.lower():
        return "invoice"
    elif "gehalt" in doc.title.lower():
        return "payslip"
    elif doc.correspondent == "Amazon":
        return "amazon_order"
    elif len(doc.content) > 10000:
        return "long_document"
    else:
        return "generic"
```

**Metadata-Rich Indexing:**
```python
# Speichere VIEL Metadata
chunk_metadata = {
    "document_id": doc.id,
    "title": doc.title,
    "correspondent": doc.correspondent,
    "tags": doc.tags,
    "created_date": doc.created,
    
    # Extracted Entities
    "amounts": [123.45, 678.90],
    "dates": ["2024-10-15"],
    "entities": ["Amazon", "DHL"],
    
    # Document Type
    "doc_type": "invoice",
    "language": "de",
    
    # Chunk Info
    "chunk_index": 0,
    "total_chunks": 1,
    "chunk_type": "full_document"  # or "section", "paragraph"
}
```

---

### 5. Advanced: Multi-Vector Retrieval (Voyage AI, Cohere)

**Konzept:**
```python
# MEHRERE Embeddings pro Dokument
class MultiVectorStore:
    def index(self, document):
        vectors = {}
        
        # 1. Summary Embedding (Überblick)
        summary = self.generate_summary(document)
        vectors["summary"] = self.embed(summary)
        
        # 2. Chunk Embeddings (Details)
        chunks = self.chunk(document)
        vectors["chunks"] = [self.embed(c) for c in chunks]
        
        # 3. Hypothetical Questions
        questions = self.generate_questions(document)
        # → Bei Gehaltsabrechnung:
        #    "Wie hoch war die Lohnsteuer?"
        #    "Was wurde an Sozialversicherung gezahlt?"
        vectors["questions"] = [self.embed(q) for q in questions]
        
        # 4. Keywords Embedding
        keywords = self.extract_keywords(document)
        vectors["keywords"] = self.embed(" ".join(keywords))
        
        return vectors

    def search(self, query):
        # Suche in ALLEN Vector-Typen
        results = {
            "summary_matches": self.search_vectors(query, type="summary"),
            "chunk_matches": self.search_vectors(query, type="chunks"),
            "question_matches": self.search_vectors(query, type="questions"),
            "keyword_matches": self.search_vectors(query, type="keywords")
        }
        
        # Kombiniere Ergebnisse mit Gewichtung
        return self.merge_results(results)
```

---

## 💡 Implementierungs-Plan für unser System

### Phase 1: Quick Wins (Sofort umsetzbar)

#### 1.1 Adaptive Chunk-Size basierend auf Dokumenttyp

```python
# src/rag/chunker.py erweitern

class AdaptiveChunker(DocumentChunker):
    def detect_document_type(self, doc):
        """Erkenne Dokumenttyp aus Metadaten"""
        title = doc.get('title', '').lower()
        correspondent = doc.get('correspondent_name', '').lower()
        tags = [t.lower() for t in doc.get('tags', [])]
        
        # Keyword-basierte Erkennung
        if any(kw in title for kw in ['rechnung', 'invoice', 'bill']):
            return 'invoice'
        elif any(kw in title for kw in ['gehalt', 'lohn', 'payslip']):
            return 'payslip'
        elif 'amazon' in correspondent:
            return 'amazon_order'
        elif any(kw in title for kw in ['vertrag', 'contract']):
            return 'contract'
        elif 'steuer' in title or 'tax' in title:
            return 'tax_document'
        else:
            return 'generic'
    
    def get_optimal_chunk_size(self, doc_type):
        """Verschiedene Chunk-Sizes pro Typ"""
        sizes = {
            'invoice': 3000,        # Volle Rechnung in 1 Chunk
            'payslip': 2500,        # Komplette Gehaltsinfo
            'amazon_order': 2000,   # Bestellung komplett
            'tax_document': 2500,   # Steuerdokumente komplett
            'contract': 1500,       # Verträge lang → kleinere Chunks
            'generic': 2000         # Default
        }
        return sizes.get(doc_type, 2000)
    
    def chunk_document(self, document):
        """Override mit adaptiver Chunk-Size"""
        doc_type = self.detect_document_type(document)
        optimal_size = self.get_optimal_chunk_size(doc_type)
        
        # Temporär neue Chunk-Size setzen
        original_size = self.chunk_size
        self.chunk_size = optimal_size
        
        # Chunken
        chunks = super().chunk_document(document)
        
        # Dokumenttyp zu Metadata hinzufügen
        for chunk in chunks:
            chunk['metadata']['document_type'] = doc_type
            chunk['metadata']['chunk_strategy'] = f'adaptive_{optimal_size}'
        
        # Restore original size
        self.chunk_size = original_size
        
        return chunks
```

**Aktivierung:**
```python
# In src/indexer.py
from src.rag.chunker import AdaptiveChunker

indexer = DocumentIndexer()
indexer.chunker = AdaptiveChunker()  # Statt Standard-Chunker
```

#### 1.2 Metadata-Extraktion für strukturierte Daten

```python
# src/rag/metadata_extractor.py (NEU)

import re
from typing import Dict, List
from datetime import datetime

class MetadataExtractor:
    """Extrahiere strukturierte Daten aus Dokumenten"""
    
    def extract_financial_data(self, content: str) -> Dict:
        """Extrahiere Geldbeträge"""
        # Regex für Euro-Beträge
        amounts = re.findall(r'(\d+[.,]\d{2})\s*€', content)
        amounts_float = [float(a.replace(',', '.')) for a in amounts]
        
        return {
            'amounts': amounts_float,
            'total_amount': sum(amounts_float) if amounts_float else None,
            'currency': 'EUR'
        }
    
    def extract_dates(self, content: str) -> List[str]:
        """Extrahiere Datumsangaben"""
        # Deutsche Datumsformate
        patterns = [
            r'\d{1,2}\.\d{1,2}\.\d{4}',  # 15.10.2024
            r'\d{4}-\d{2}-\d{2}',         # 2024-10-15
        ]
        
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, content))
        
        return list(set(dates))
    
    def extract_month_year(self, content: str) -> Dict:
        """Extrahiere Monat und Jahr (für Gehaltsabrechnungen)"""
        months_de = {
            'januar': 1, 'februar': 2, 'märz': 3, 'april': 4,
            'mai': 5, 'juni': 6, 'juli': 7, 'august': 8,
            'september': 9, 'oktober': 10, 'november': 11, 'dezember': 12
        }
        
        content_lower = content.lower()
        
        # Finde Monat
        month = None
        for month_name, month_num in months_de.items():
            if month_name in content_lower:
                month = month_num
                break
        
        # Finde Jahr
        years = re.findall(r'\b(20\d{2})\b', content)
        year = int(years[0]) if years else None
        
        return {
            'month': month,
            'year': year,
            'period': f"{month}/{year}" if month and year else None
        }
    
    def extract_tax_info(self, content: str) -> Dict:
        """Extrahiere Steuerinformationen"""
        tax_keywords = {
            'lohnsteuer': r'lohnsteuer[:\s]+(\d+[.,]\d{2})',
            'solidaritätszuschlag': r'solidaritätszuschlag[:\s]+(\d+[.,]\d{2})',
            'kirchensteuer': r'kirchensteuer[:\s]+(\d+[.,]\d{2})'
        }
        
        taxes = {}
        for tax_type, pattern in tax_keywords.items():
            match = re.search(pattern, content.lower())
            if match:
                amount = float(match.group(1).replace(',', '.'))
                taxes[tax_type] = amount
        
        return taxes
    
    def extract_all_metadata(self, document: Dict) -> Dict:
        """Komplette Metadaten-Extraktion"""
        content = document.get('content', '')
        
        metadata = {
            **self.extract_financial_data(content),
            'dates': self.extract_dates(content),
            **self.extract_month_year(content),
            'tax_info': self.extract_tax_info(content),
            'has_amounts': bool(self.extract_financial_data(content)['amounts']),
            'content_length': len(content),
            'word_count': len(content.split())
        }
        
        return metadata
```

**Integration:**
```python
# In src/rag/chunker.py
from src.rag.metadata_extractor import MetadataExtractor

class EnhancedChunker(AdaptiveChunker):
    def __init__(self):
        super().__init__()
        self.metadata_extractor = MetadataExtractor()
    
    def chunk_document(self, document):
        # 1. Extrahiere strukturierte Metadaten
        extracted = self.metadata_extractor.extract_all_metadata(document)
        
        # 2. Normale Chunks erstellen
        chunks = super().chunk_document(document)
        
        # 3. Füge extrahierte Metadaten zu JEDEM Chunk hinzu
        for chunk in chunks:
            chunk['metadata'].update({
                'extracted_data': extracted,
                'searchable_amounts': extracted.get('amounts', []),
                'searchable_period': extracted.get('period'),
                'has_tax_info': bool(extracted.get('tax_info'))
            })
        
        return chunks
```

#### 1.3 Verbesserte Query-Detection

```python
# src/rag/query_analyzer.py (NEU)

class QueryAnalyzer:
    """Analysiere User-Query für besseres Retrieval"""
    
    def detect_query_type(self, query: str) -> str:
        """Erkenne Query-Typ"""
        query_lower = query.lower()
        
        # Analytical
        if any(kw in query_lower for kw in ['alle', 'übersicht', 'liste', 'tabelle', 'summiere']):
            return 'analytical'
        
        # Financial
        if any(kw in query_lower for kw in ['kosten', 'preis', 'betrag', 'summe', 'euro']):
            return 'financial'
        
        # Temporal
        if any(kw in query_lower for kw in ['monat', 'jahr', '2024', '2023', 'letzte', 'diese']):
            return 'temporal'
        
        # Simple lookup
        return 'simple'
    
    def extract_filters(self, query: str) -> Dict:
        """Extrahiere Filter aus Query"""
        filters = {}
        
        # Correspondent
        correspondent_match = re.search(r'bei\s+(\w+)|von\s+(\w+)|at\s+(\w+)', query, re.I)
        if correspondent_match:
            correspondent = next(g for g in correspondent_match.groups() if g)
            filters['correspondent'] = correspondent.capitalize()
        
        # Monat
        months_de = {
            'januar': 1, 'februar': 2, 'märz': 3, 'april': 4, 'mai': 5,
            'juni': 6, 'juli': 7, 'august': 8, 'september': 9,
            'oktober': 10, 'november': 11, 'dezember': 12
        }
        for month_name, month_num in months_de.items():
            if month_name in query.lower():
                filters['month'] = month_num
                break
        
        # Jahr
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            filters['year'] = int(year_match.group(1))
        
        # Dokumenttyp
        if 'rechnung' in query.lower():
            filters['doc_type'] = 'invoice'
        elif 'gehalt' in query.lower():
            filters['doc_type'] = 'payslip'
        
        return filters
    
    def suggest_n_results(self, query_type: str) -> int:
        """Empfohlene Anzahl Ergebnisse pro Query-Typ"""
        suggestions = {
            'analytical': 30,   # Viele Dokumente
            'financial': 20,    # Mehrere Dokumente
            'temporal': 25,     # Zeitraum = mehrere Docs
            'simple': 10        # Standard
        }
        return suggestions.get(query_type, 10)
```

---

### Phase 2: Mittelfristig (Nächste Iteration)

#### 2.1 Hypothetical Question Generation

```python
# Generiere typische Fragen zu Dokumenten
class QuestionGenerator:
    def generate_for_payslip(self, doc):
        month = doc.get('month')
        year = doc.get('year')
        
        return [
            f"Wie hoch war die Lohnsteuer {month}/{year}?",
            f"Was habe ich {month}/{year} verdient?",
            f"Sozialversicherung {month}/{year}",
            f"Netto-Gehalt {month}/{year}"
        ]
    
    def generate_for_invoice(self, doc):
        correspondent = doc.get('correspondent')
        
        return [
            f"Rechnung von {correspondent}",
            f"Wie viel habe ich bei {correspondent} bezahlt?",
            f"Bestellung {correspondent}"
        ]
```

#### 2.2 Hierarchical Chunking

```python
# Parent-Child Beziehungen
class HierarchicalChunker:
    def chunk(self, document):
        # Parent: Vollständiges Dokument (Summary)
        parent = {
            'text': document['content'][:8000],  # Erste 8K tokens
            'type': 'parent',
            'summary': self.generate_summary(document)
        }
        
        # Children: Detail-Chunks
        children = self.recursive_chunk(document['content'])
        
        # Link Children → Parent
        for child in children:
            child['parent_id'] = parent['id']
        
        return {
            'parent': parent,
            'children': children
        }
```

#### 2.3 Structured Data Store (SQLite)

```python
# Separater Store für strukturierte Daten
class StructuredDataStore:
    def __init__(self):
        self.db = sqlite3.connect('structured_data.db')
        self.create_tables()
    
    def create_tables(self):
        # Tabelle für Gehaltsabrechnungen
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS payslips (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                month INTEGER,
                year INTEGER,
                gross_salary REAL,
                net_salary REAL,
                income_tax REAL,
                social_security REAL,
                created_date DATE
            )
        ''')
        
        # Tabelle für Rechnungen
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY,
                document_id INTEGER,
                correspondent TEXT,
                total_amount REAL,
                invoice_date DATE,
                created_date DATE
            )
        ''')
    
    def query_structured(self, sql):
        """SQL-Query auf strukturierte Daten"""
        # "SELECT SUM(income_tax) FROM payslips WHERE year=2017"
        return self.db.execute(sql).fetchall()
```

---

### Phase 3: Advanced (Später)

#### 3.1 Multi-Vector Indexing
#### 3.2 LLM-based Document Summarization
#### 3.3 Cross-Document Linking
#### 3.4 Automatic Query Routing (Structured vs Semantic)

---

## 🎯 Zusammenfassung

### Was NotebookLM & Co. besser machen:

1. **Adaptive Chunking** statt fixer Größe
2. **Hierarchical Indexing** (Parent-Child)
3. **Metadata-First** Ansatz
4. **Multi-Vector Retrieval**
5. **Structured Data Extraction**
6. **Large Context Windows** (Claude: 200K)

### Was wir umsetzen können:

**Sofort:**
- ✅ Adaptive Chunk-Size nach Dokumenttyp
- ✅ Metadata-Extraktion (Beträge, Daten, etc.)
- ✅ Query-Type Detection

**Bald:**
- 🔄 Hypothetical Questions
- 🔄 Parent-Child Chunks
- 🔄 Structured Data Store (SQLite)

**Später:**
- ⏳ Multi-Vector Indexing
- ⏳ Cross-Document Linking
- ⏳ LLM Summarization

---

## 📚 Weitere Ressourcen

- [Google NotebookLM Blog](https://blog.google/technology/ai/notebooklm-gemini-pro-update/)
- [OpenAI Assistants API](https://platform.openai.com/docs/assistants/overview)
- [Anthropic Extended Context](https://www.anthropic.com/index/claude-2-1-prompting)
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [LangChain Parent-Child Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
