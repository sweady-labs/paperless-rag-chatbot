"""
German Synonym Expansion for RAG Queries.

Expands German queries with common synonyms to improve recall.
Particularly useful for documents with varying terminology.

Example:
    "Rechnung" → "Rechnung Beleg Quittung Faktura"
    "Arzt" → "Arzt Doktor Mediziner"
"""

import re
from typing import List, Dict

# German synonym dictionary
# Format: {term: [synonym1, synonym2, ...]}
GERMAN_SYNONYMS: Dict[str, List[str]] = {
    # Documents & paperwork
    'rechnung': ['beleg', 'quittung', 'faktura', 'invoice'],
    'dokument': ['unterlage', 'schriftstück', 'papier'],
    'bescheinigung': ['nachweis', 'bestätigung', 'attest', 'zertifikat'],
    'vertrag': ['vereinbarung', 'kontrakt', 'abkommen'],
    'brief': ['schreiben', 'post', 'anschreiben'],
    'formular': ['vordruck', 'bogen', 'antrag'],
    
    # Medical
    'arzt': ['doktor', 'mediziner', 'doc'],
    'arztbrief': ['befund', 'medizinischer bericht', 'ärztlicher bericht'],
    'krankenhaus': ['klinik', 'hospital', 'spital'],
    'behandlung': ['therapie', 'untersuchung', 'konsultation'],
    'medikament': ['arzneimittel', 'präparat', 'tablette'],
    'rezept': ['verschreibung', 'verordnung'],
    'überweisung': ['zuweisung'],  # medical referral
    'diagnose': ['befund', 'krankheitsbild'],
    'patient': ['erkrankter', 'kranker'],
    
    # Financial
    'bezahlung': ['zahlung', 'überweisung', 'transaktion'],
    'kosten': ['preis', 'betrag', 'gebühr', 'ausgabe'],
    'versicherung': ['assekuranz', 'police'],
    'steuer': ['abgabe', 'tribut'],
    'gehalt': ['lohn', 'verdienst', 'einkommen', 'bezüge'],
    'gehaltsabrechnung': ['entgeltabrechnung', 'lohnabrechnung', 'lohnzettel'],
    'entgeltabrechnung': ['gehaltsabrechnung', 'lohnabrechnung'],
    'konto': ['bankkonto', 'girokonto'],
    'kontoauszug': ['bankauszug', 'statement', 'kontoblatt'],
    'mahnung': ['zahlungserinnerung', 'inkasso'],
    'kredit': ['darlehen', 'finanzierung'],
    
    # Utilities
    'strom': ['elektrizität', 'energie'],
    'wasser': ['trinkwasser'],
    'gas': ['erdgas'],
    'miete': ['pacht'],
    
    # Institutions
    'behörde': ['amt', 'verwaltung', 'dienststelle'],
    'firma': ['unternehmen', 'betrieb', 'gesellschaft'],
    'schule': ['bildungseinrichtung'],
    
    # Time-related
    'termin': ['appointment', 'verabredung', 'datum'],
    'frist': ['deadline', 'stichtag'],
    
    # Actions
    'antrag': ['gesuch', 'anfrage', 'ersuchen'],
    'genehmigung': ['erlaubnis', 'zustimmung', 'bewilligung'],
    'änderung': ['modifikation', 'anpassung', 'korrektur'],
    'kündigung': ['auflösung', 'beendigung'],
}


def expand_query_with_synonyms(query: str, max_synonyms: int = 2) -> str:
    """
    Expand query with German synonyms.
    
    Args:
        query: Original query string
        max_synonyms: Maximum synonyms to add per term (default: 2)
        
    Returns:
        Expanded query with synonyms
        
    Example:
        >>> expand_query_with_synonyms("Rechnung von Arzt")
        "Rechnung Beleg Quittung von Arzt Doktor"
    """
    words = query.lower().split()
    expanded_parts = []
    
    for word in words:
        # Clean word (remove punctuation)
        clean_word = re.sub(r'[^\w]', '', word)
        
        # Add original word
        expanded_parts.append(word)
        
        # Add synonyms if available
        if clean_word in GERMAN_SYNONYMS:
            synonyms = GERMAN_SYNONYMS[clean_word][:max_synonyms]
            expanded_parts.extend(synonyms)
    
    return ' '.join(expanded_parts)


def get_synonyms(term: str) -> List[str]:
    """
    Get synonyms for a German term.
    
    Args:
        term: German term
        
    Returns:
        List of synonyms (empty if none found)
    """
    clean_term = re.sub(r'[^\w]', '', term.lower())
    return GERMAN_SYNONYMS.get(clean_term, [])


def add_custom_synonyms(term: str, synonyms: List[str]):
    """
    Add custom synonyms to the dictionary.
    
    Args:
        term: Base term
        synonyms: List of synonyms to add
    """
    clean_term = re.sub(r'[^\w]', '', term.lower())
    
    if clean_term in GERMAN_SYNONYMS:
        # Add to existing
        GERMAN_SYNONYMS[clean_term].extend(synonyms)
        # Remove duplicates
        GERMAN_SYNONYMS[clean_term] = list(set(GERMAN_SYNONYMS[clean_term]))
    else:
        # Create new entry
        GERMAN_SYNONYMS[clean_term] = synonyms


def should_expand_query(query: str) -> bool:
    """
    Determine if a query should be expanded with synonyms.
    
    Only expand simple keyword queries, not complex questions or queries with proper nouns/names.
    
    Args:
        query: Query string
        
    Returns:
        True if query should be expanded
    """
    # Check for proper nouns BEFORE lowercasing
    original_words = query.split()
    for i, word in enumerate(original_words):
        if i > 0 and word and word[0].isupper():  # Capitalized word not at start
            return False
    
    words = query.lower().split()
    
    # Don't expand empty queries
    if len(words) == 0:
        return False
    
    # Don't expand very long analytical queries
    if len(words) > 10:
        return False
    
    # Don't expand questions (start with question words)
    question_words = ['was', 'wie', 'wann', 'wo', 'wer', 'warum', 'welche', 'welcher', 'welches', 
                      'what', 'how', 'when', 'where', 'who', 'why', 'which']
    if any(query.lower().startswith(qw) for qw in question_words):
        return False
    
    # Don't expand if contains analytical keywords
    analytical_keywords = ['alle', 'gesamt', 'liste', 'übersicht', 'tabelle', 'how many', 'wieviele', 
                          'zeige', 'show', 'display']
    if any(kw in query.lower() for kw in analytical_keywords):
        return False
    
    # Only expand if contains at least one synonymizable term
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in GERMAN_SYNONYMS:
            return True
    
    return False
    
    # Don't expand long analytical queries
    if len(words) > 10:
        return False
    
    # Don't expand if contains analytical keywords
    analytical_keywords = ['alle', 'gesamt', 'liste', 'übersicht', 'tabelle', 'how many', 'wieviele']
    if any(kw in query.lower() for kw in analytical_keywords):
        return False
    
    # Only expand if contains at least one synonymizable term
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in GERMAN_SYNONYMS:
            return True
    
    return False


# Precompute common expanded queries (optimization)
_COMMON_EXPANSIONS = {
    'rechnung': 'rechnung beleg quittung',
    'arzt': 'arzt doktor mediziner',
    'krankenhaus': 'krankenhaus klinik hospital',
    'lohn': 'lohn gehalt verdienst',
    'strom': 'strom elektrizität energie',
    'miete': 'miete pacht',
}


def quick_expand(term: str) -> str:
    """
    Quick expansion using precomputed common terms.
    
    Args:
        term: Single term to expand
        
    Returns:
        Expanded string or original term
    """
    clean_term = re.sub(r'[^\w]', '', term.lower())
    return _COMMON_EXPANSIONS.get(clean_term, term)


if __name__ == "__main__":
    # Test expansions
    test_queries = [
        "Rechnung von Arzt",
        "Lohnsteuerbescheinigung",
        "Stromrechnung 2024",
        "Vertrag mit Krankenhaus",
        "Alle Dokumente",  # Should not expand (analytical)
    ]
    
    print("German Synonym Expansion Test")
    print("=" * 60)
    print()
    
    for query in test_queries:
        should_expand = should_expand_query(query)
        expanded = expand_query_with_synonyms(query) if should_expand else query
        
        if should_expand:
            print(f"Original:  {query}")
            print(f"Expanded:  {expanded}")
            print()
        else:
            print(f"Skip:      {query} (no expansion needed)")
            print()
