#!/usr/bin/env python3
"""
Test synonym expansion functionality.

Tests the synonym expansion module to ensure it:
1. Correctly identifies queries that should be expanded
2. Expands German queries with appropriate synonyms
3. Doesn't expand unsuitable queries (questions, complex queries)
4. Improves recall for document retrieval

Usage:
    python test_synonyms.py
"""

import sys
import logging
from src.rag.synonyms import expand_query_with_synonyms, should_expand_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_should_expand():
    """Test the should_expand_query() function."""
    print("\n" + "="*80)
    print("TEST 1: Query Expansion Detection")
    print("="*80)
    
    # Should expand (short, keyword-based)
    should_expand_queries = [
        "Rechnung",
        "Vertrag bei Telekom",
        "Dokument von VW",
        "Arztbrief"
    ]
    
    # Should NOT expand (questions, complex queries)
    should_not_expand = [
        "Was sind alle Rechnungen von 2024?",
        "Wie hoch ist die Summe aller Ausgaben?",
        "Zeige mir die neuesten Dokumente",
        "Wann wurde der Vertrag unterschrieben?"
    ]
    
    print("\nQueries that SHOULD be expanded:")
    for query in should_expand_queries:
        result = should_expand_query(query)
        status = "✓" if result else "✗"
        print(f"  {status} '{query}' -> {result}")
    
    print("\nQueries that should NOT be expanded:")
    for query in should_not_expand:
        result = should_expand_query(query)
        status = "✓" if not result else "✗"
        print(f"  {status} '{query}' -> {result}")


def test_synonym_expansion():
    """Test the expand_query_with_synonyms() function."""
    print("\n" + "="*80)
    print("TEST 2: Synonym Expansion")
    print("="*80)
    
    test_cases = [
        # Document-related
        ("Rechnung", ["Rechnung", "Faktura", "Beleg", "Invoice"]),
        ("Vertrag", ["Vertrag", "Kontrakt", "Vereinbarung", "Contract"]),
        ("Dokument", ["Dokument", "Unterlage", "Schriftstück", "Document"]),
        
        # Medical
        ("Arztbrief", ["Arztbrief", "Befund", "medizinischer Bericht"]),
        ("Rezept", ["Rezept", "Verschreibung", "Verordnung"]),
        
        # Financial
        ("Kontoauszug", ["Kontoauszug", "Bankauszug", "Statement"]),
        ("Bescheinigung", ["Bescheinigung", "Nachweis", "Bestätigung"]),
        
        # Insurance
        ("Versicherung", ["Versicherung", "Police", "Versicherungspolice"]),
        
        # No expansion (no synonyms)
        ("Telefonnummer", None),  # Should return original
    ]
    
    for original, expected_terms in test_cases:
        expanded = expand_query_with_synonyms(original)
        
        print(f"\nOriginal: '{original}'")
        print(f"Expanded: '{expanded}'")
        
        if expected_terms:
            # Check if all expected terms are in the expanded query
            missing = [term for term in expected_terms if term not in expanded]
            if missing:
                print(f"  ✗ MISSING: {missing}")
            else:
                print(f"  ✓ Contains all expected synonyms")
        else:
            # Should return original if no synonyms
            if expanded == original:
                print(f"  ✓ No expansion (as expected)")
            else:
                print(f"  ✗ Unexpected expansion!")


def test_integration():
    """Test the full workflow: should_expand + expand."""
    print("\n" + "="*80)
    print("TEST 3: Integration Test (should_expand + expand)")
    print("="*80)
    
    queries = [
        "Rechnung",
        "Vertrag bei Telekom",
        "Was sind alle Rechnungen?",
        "Arztbrief von Dr. Müller",
        "Zeige mir alle Dokumente"
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        
        if should_expand_query(query):
            expanded = expand_query_with_synonyms(query)
            print(f"   ✓ Expanded: '{expanded}'")
        else:
            print(f"   ⊘ No expansion (query too complex)")


def test_real_world_scenarios():
    """Test real-world query scenarios."""
    print("\n" + "="*80)
    print("TEST 4: Real-World Scenarios")
    print("="*80)
    
    scenarios = [
        {
            'description': 'Simple invoice search',
            'query': 'Rechnung',
            'should_expand': True,
            'expected_improvement': 'Should find invoices labeled as "Faktura" or "Beleg"'
        },
        {
            'description': 'Contract with company filter',
            'query': 'Vertrag bei Telekom',
            'should_expand': True,
            'expected_improvement': 'Should find contracts also labeled as "Kontrakt" or "Vereinbarung"'
        },
        {
            'description': 'Medical report search',
            'query': 'Arztbrief',
            'should_expand': True,
            'expected_improvement': 'Should find medical reports labeled as "Befund" or "medizinischer Bericht"'
        },
        {
            'description': 'Complex analytical query',
            'query': 'Was sind alle Rechnungen von 2024?',
            'should_expand': False,
            'expected_improvement': 'No expansion - LLM should handle the semantic meaning'
        },
        {
            'description': 'Question about totals',
            'query': 'Wie hoch ist die Gesamtsumme?',
            'should_expand': False,
            'expected_improvement': 'No expansion - this is a calculation question'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"Scenario: {scenario['description']}")
        print(f"Query: '{scenario['query']}'")
        
        should_expand = should_expand_query(scenario['query'])
        
        if should_expand == scenario['should_expand']:
            print(f"✓ Expansion decision: {'YES' if should_expand else 'NO'} (correct)")
        else:
            print(f"✗ Expansion decision: {'YES' if should_expand else 'NO'} (WRONG - expected {'YES' if scenario['should_expand'] else 'NO'})")
        
        if should_expand:
            expanded = expand_query_with_synonyms(scenario['query'])
            print(f"Expanded: '{expanded}'")
        
        print(f"Expected improvement: {scenario['expected_improvement']}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SYNONYM EXPANSION TEST SUITE")
    print("="*80)
    
    try:
        test_should_expand()
        test_synonym_expansion()
        test_integration()
        test_real_world_scenarios()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS COMPLETED")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the expansion results above")
        print("2. Test with actual queries: make serve-all")
        print("3. Compare recall before/after enabling ENABLE_SYNONYM_EXPANSION")
        print("4. Monitor logs for: 'Synonym expansion: ...'")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
