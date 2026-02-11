"""
Advanced Examples: Dependency Parsing and Word Embeddings
Demonstrates the new semantic search features.
"""

import logging
logging.basicConfig(level=logging.INFO)

# Example 1: Dependency Parsing Demo
def demo_dependency_parsing():
    """Demonstrate dependency parsing capabilities."""
    print("=" * 80)
    print("EXAMPLE 1: DEPENDENCY PARSING")
    print("=" * 80)
    
    from src.dependency_parser import DependencyParser
    
    parser = DependencyParser()
    
    test_queries = [
        "bike with child seat",
        "bike for child",
        "lightweight mountain bike for hills",
        "carbon fiber frame road bike under 1000 euro",
        "running shoes for marathon training"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        # Analyze query structure
        analysis = parser.analyze_query_structure(query)
        
        print(f"Noun Phrases: {analysis['noun_phrases']}")
        print(f"Compound Terms: {analysis['compound_terms']}")
        print(f"Key Concepts: {analysis['key_concepts']}")
        
        if analysis['relationships']:
            print("Relationships:")
            for rel in analysis['relationships']:
                print(f"  • {rel['head']} {rel['prep']} {rel['object']}")
        
        # Enhanced query
        enhanced = parser.enhance_query_for_search(query)
        print(f"Enhanced Query: '{enhanced}'")


# Example 2: Word Embeddings Demo
def demo_word_embeddings():
    """Demonstrate word embeddings for semantic similarity."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: WORD EMBEDDINGS & SEMANTIC SIMILARITY")
    print("=" * 80)
    
    from src.word_embeddings import SemanticEmbeddings
    
    embeddings = SemanticEmbeddings()
    
    # Test queries
    test_queries = [
        "cheap bike for hills",
        "affordable mountain bike",
        "lightweight road bike for racing"
    ]
    
    # Test documents
    test_docs = [
        "Budget mountain bike perfect for hill climbing and steep terrain",
        "Professional racing road bike with carbon fiber frame",
        "Affordable hybrid bike for city commuting",
        "Premium MTB for off-road trails and mountains",
        "Economical bike suitable for climbing hills"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        
        # Query expansion
        expansions = embeddings.expand_query(query)
        print(f"Semantic Expansions: {expansions}")
        
        # Enhanced query
        enhanced = embeddings.enhance_query_with_embeddings(query, max_additions=5)
        print(f"Enhanced Query: '{enhanced}'")
        
        # Compute similarities with documents
        similarities = embeddings.compute_similarity(query, test_docs, top_k=3)
        print("\nTop Matching Documents:")
        for idx, score in similarities:
            print(f"  • Score {score:.3f}: {test_docs[idx][:70]}...")


# Example 3: Full Search Engine with Advanced Features
def demo_advanced_search():
    """Demonstrate full search engine with dependency parsing and embeddings."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 3: ADVANCED SEMANTIC SEARCH")
    print("=" * 80)
    
    from src.search_engine import ProductSearchEngine
    
    # Initialize search engine with advanced features
    print("\nInitializing search engine with advanced features...")
    engine = ProductSearchEngine(
        use_elasticsearch=True,
        use_dependency_parsing=True,  # Enable dependency parsing
        use_embeddings=True,           # Enable word embeddings
        hybrid_alpha=0.6               # 60% TF-IDF, 40% embeddings
    )
    
    # Load sample data
    print("Loading sample product data...")
    import pandas as pd
    
    # Create sample bike products
    sample_products = pd.DataFrame([
        {
            "name": "Budget Mountain Bike MTB-200",
            "brand": "Drag",
            "desc": "Perfect for hill climbing and steep terrain. Affordable price.",
            "price_numeric": 450,
            "category": "MTB",
            "weight": 14,
            "rating": 4.2,
            "reviews_count": 150,
            "availability": "In Stock"
        },
        {
            "name": "Carbon Road Bike Pro-Speed",
            "brand": "Trek",
            "desc": "Professional racing bike with carbon fiber frame. Lightweight design.",
            "price_numeric": 1800,
            "category": "Road",
            "weight": 7.5,
            "rating": 4.8,
            "reviews_count": 320,
            "availability": "In Stock"
        },
        {
            "name": "Family Bike with Child Seat",
            "brand": "Schwinn",
            "desc": "City bike with integrated child seat. Perfect for family rides.",
            "price_numeric": 320,
            "category": "City",
            "weight": 16,
            "rating": 4.5,
            "reviews_count": 89,
            "availability": "In Stock"
        },
        {
            "name": "Hybrid Commuter Bike Urban-X",
            "brand": "Giant",
            "desc": "Versatile bike for city commuting and light trails.",
            "price_numeric": 550,
            "category": "Hybrid",
            "weight": 12,
            "rating": 4.3,
            "reviews_count": 210,
            "availability": "In Stock"
        }
    ])
    
    engine.df = sample_products
    
    # Build index
    print("Building search index with semantic enrichment...")
    engine.build_index()
    
    # Test searches with advanced features
    test_searches = [
        {
            "query": "cheap bike for hills",
            "description": "Semantic expansion: 'cheap' → 'budget', 'affordable'; 'hills' → 'mountain', 'climbing'"
        },
        {
            "query": "bike with child seat",
            "description": "Dependency parsing: understands 'bike WITH child seat' relationship"
        },
        {
            "query": "lightweight bike for racing",
            "description": "Semantic matching: 'lightweight' + 'racing' → carbon road bikes"
        }
    ]
    
    for test in test_searches:
        query = test["query"]
        print(f"\n\n{'='*80}")
        print(f"Search: '{query}'")
        print(f"Expected: {test['description']}")
        print("="*80)
        
        # Search with verbose output
        results = engine.search(query, top_k=3, verbose=True)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['product']}")
                print(f"   Brand: {result['brand']}")
                print(f"   Price: ${result['price']:.2f}")
                print(f"   Score: {result['score']:.4f}")
                if 'category' in result:
                    print(f"   Category: {result['category']}")
        else:
            print("No results found")
    
    print("\n\n" + "="*80)
    print("Advanced search demonstration complete!")
    print("="*80)


# Example 4: Hybrid Scoring Comparison
def demo_hybrid_scoring():
    """Compare standard vs hybrid scoring."""
    print("\n\n" + "=" * 80)
    print("EXAMPLE 4: HYBRID SCORING COMPARISON")
    print("=" * 80)
    
    from src.word_embeddings import SemanticEmbeddings
    
    embeddings = SemanticEmbeddings()
    
    query = "affordable bike for mountain trails"
    
    documents = [
        "Budget MTB for off-road and hills",  # Good semantic match
        "Mountain bike trails guide book",     # Keyword match but wrong item
        "Economical bicycle for rough terrain", # Synonym match
        "Expensive professional racing bike"   # Poor match
    ]
    
    print(f"\nQuery: '{query}'")
    print("\nDocuments:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    print("\nSemantic Similarity Scores:")
    similarities = embeddings.compute_similarity(query, documents)
    for idx, score in similarities:
        print(f"  Doc {idx+1}: {score:.3f}")
    
    print("\nObservation:")
    print("  - Document 1 scores high (budget + mountain + trails)")
    print("  - Document 2 has keywords but wrong context")
    print("  - Document 3 scores high (synonyms: affordable→economical)")
    print("  - Document 4 scores low (opposite: expensive vs affordable)")


# Main execution
if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║           ADVANCED SEMANTIC SEARCH - FEATURE DEMONSTRATIONS                  ║
║                                                                              ║
║  This script demonstrates the new advanced features:                         ║
║    1. Dependency Parsing - Understanding phrase relationships                ║
║    2. Word Embeddings - Semantic similarity and query expansion              ║
║    3. Full Search Engine - Integrated advanced search                        ║
║    4. Hybrid Scoring - Combining TF-IDF with embeddings                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\n   NOTE: Make sure you have installed the required dependencies:")
    print("   pip install spacy sentence-transformers gensim")
    print("   python -m spacy download en_core_web_sm")
    print("\n" + "="*80)
    
    try:
        # Run all demonstrations
        demo_dependency_parsing()
        demo_word_embeddings()
        demo_hybrid_scoring()
        
        # Optional: Full search demo (requires Elasticsearch)
        print("\n\n" + "="*80)
        response = input("Run full search engine demo? (requires Elasticsearch) [y/N]: ")
        if response.lower() == 'y':
            demo_advanced_search()
        else:
            print("Skipping full search demo.")
        
        print("\n\nAll demonstrations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
