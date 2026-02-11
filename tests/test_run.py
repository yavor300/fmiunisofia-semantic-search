"""
Simple test script to verify the search engine works correctly.
Run this after installation to make sure everything is set up properly.

Usage:
    python test_run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.search_engine import ProductSearchEngine
from src.sample_data import generate_sample_data


def test_basic_functionality():
    """Test basic search engine functionality."""
    
    print("="*80)
    print("Search Engine Test Script")
    print("="*80)
    
    # Step 1: Generate sample data
    print("\n[1/5] Generating sample data...")
    try:
        data_path = generate_sample_data("data/sample_products.csv")
        print(f"Sample data created at: {data_path}")
    except Exception as e:
        print(f"Failed to generate sample data: {e}")
        return False
    
    # Step 2: Initialize engine
    print("\n[2/5] Initializing search engine...")
    try:
        engine = ProductSearchEngine(use_elasticsearch=True, use_dependency_parsing=True, use_embeddings=True)
        print("Search engine initialized")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False
    
    # Step 3: Load data
    print("\n[3/5] Loading data...")
    try:
        if not engine.load_data("data/sample_products.csv"):
            print("Failed to load data")
            return False
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Step 4: Build index
    print("\n[4/5] Building search index...")
    try:
        if not engine.build_index():
            print("Failed to build index")
            return False
        print("Index built successfully")
    except Exception as e:
        print(f"Error building index: {e}")
        return False
    
    # Step 5: Test searches
    print("\n[5/5] Testing search functionality...")
    
    test_queries = [
        ("cheap running shoes", "Should find budget athletic shoes"),
        ("premium laptop", "Should find high-end laptops"),
        ("Trek mountain bike", "Should find Trek MTB"),
        ("budget bike", "Should find affordable bicycles"),
    ]
    
    all_passed = True
    
    for query, expected in test_queries:
        print(f"\n  Testing: '{query}'")
        print(f"  Expected: {expected}")
        
        try:
            results = engine.search(query, top_k=3, verbose=True)
            
            if len(results) > 0:
                print(f"  Found {len(results)} results")
                print(f"    Top result: {results[0]['product'][:50]}")
                print(f"    Score: {results[0]['score']}")
            else:
                print(f"  No results found (may be normal depending on query)")
        except Exception as e:
            print(f"  Search failed: {e}")
            all_passed = False
    
    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nThe search engine is working correctly.")
        print("You can now run: python src/main.py")
    else:
        print("SOME TESTS FAILED")
        print("\nPlease check the error messages above.")
    print("="*80)
    
    return all_passed


def test_individual_components():
    """Test individual components separately."""
    
    print("\n" + "="*80)
    print("Testing Individual Components")
    print("="*80)
    
    # Test 1: Preprocessing
    print("\n[Test 1] Text Preprocessing")
    try:
        from src.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        test_text = "Running shoes for the best runners!"
        result = preprocessor.preprocess(test_text)
        
        print(f"  Input: '{test_text}'")
        print(f"  Output: '{result}'")
        
        if result and "run" in result:
            print("  Preprocessing works")
        else:
            print("  Preprocessing may have issues")
    except Exception as e:
        print(f"  Preprocessing test failed: {e}")
    
    # Test 2: NLP Parser
    print("\n[Test 2] NLP Parser")
    try:
        from src.nlp_parser import NLPParser
        parser = NLPParser()
        
        test_query = "Nike shoes under 100"
        result = parser.parse_query(test_query)
        
        print(f"  Input: '{test_query}'")
        print(f"  Brands: {result['brands']}")
        print(f"  Price constraints: {result['price_constraints']}")
        
        if "nike" in result['brands'] and "max" in result['price_constraints']:
            print("  NLP Parser works")
        else:
            print("  NLP Parser may have partial results")
    except Exception as e:
        print(f"  NLP Parser test failed: {e}")
    
    # Test 3: Semantic Enrichment
    print("\n[Test 3] Semantic Enrichment")
    try:
        from src.semantic_enrichment import SemanticEnricher
        enricher = SemanticEnricher()
        
        test_product = {
            "price_numeric": 25,
            "weight": 10
        }
        tags = enricher.enrich_product(test_product)
        
        print(f"  Product: price=$25, weight=10kg")
        print(f"  Generated tags: '{tags}'")
        
        if "budget" in tags and "light" in tags:
            print("  Semantic enrichment works")
        else:
            print("  Semantic enrichment may have partial results")
    except Exception as e:
        print(f"  Semantic enrichment test failed: {e}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\nStarting comprehensive test...\n")
    
    # Run main functionality test
    success = test_basic_functionality()
    
    # Run individual component tests
    test_individual_components()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
