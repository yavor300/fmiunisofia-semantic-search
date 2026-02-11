"""
Example usage scenarios for the search engine.
This script demonstrates various ways to use the system.
"""

from src.search_engine import ProductSearchEngine
from src.sample_data import generate_sample_data


def example_1_basic_search():
    """Example 1: Basic search with sample data."""
    
    print("="*80)
    print("Example 1: Basic Search")
    print("="*80)
    
    # Generate and load sample data
    data_path = generate_sample_data("data/sample_products.csv")
    
    engine = ProductSearchEngine()
    engine.load_data(data_path)
    engine.build_index()
    
    # Simple search
    query = "running shoes"
    print(f"\nSearching for: '{query}'")
    results = engine.search(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['product']}")
        print(f"   Brand: {result['brand']}, Price: ${result['price']}")
        print(f"   Relevance: {result['score']:.4f}")


def example_2_price_filtering():
    """Example 2: Search with price constraints."""
    
    print("\n" + "="*80)
    print("Example 2: Price-Constrained Search")
    print("="*80)
    
    engine = ProductSearchEngine()
    engine.load_data("data/sample_products.csv")
    engine.build_index()
    
    queries = [
        "laptop under 700",
        "bike between 500 and 1000",
        "premium headphones over 300"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, top_k=2, verbose=True)
        
        if results:
            for result in results:
                print(f"  → {result['product'][:50]} - ${result['price']}")


def example_3_brand_search():
    """Example 3: Brand-specific search."""
    
    print("\n" + "="*80)
    print("Example 3: Brand-Specific Search")
    print("="*80)
    
    engine = ProductSearchEngine()
    engine.load_data("data/sample_products.csv")
    engine.build_index()
    
    queries = [
        "Nike running shoes",
        "Trek mountain bike",
        "Dell laptop"
    ]
    
    for query in queries:
        print(f"\nSearching: '{query}'")
        results = engine.search(query, top_k=3)
        
        for result in results:
            print(f"  • {result['product'][:40]} ({result['brand']}) - ${result['price']}")


def example_4_semantic_search():
    """Example 4: Semantic search using enriched tags."""
    
    print("\n" + "="*80)
    print("Example 4: Semantic Search (Budget/Premium/Light)")
    print("="*80)
    
    engine = ProductSearchEngine()
    engine.load_data("data/sample_products.csv")
    engine.build_index()
    
    # These queries work because of semantic enrichment
    # The words "budget", "premium", "light" are added as tags
    queries = [
        "budget bike",
        "premium laptop",
        "affordable shoes",
        "light mountain bike"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("(Note: These terms are added via semantic enrichment)")
        results = engine.search(query, top_k=2)
        
        for result in results:
            print(f"  → {result['product'][:45]} - ${result['price']} (score: {result['score']:.3f})")


def example_5_complex_queries():
    """Example 5: Complex multi-constraint queries."""
    
    print("\n" + "="*80)
    print("Example 5: Complex Multi-Constraint Queries")
    print("="*80)
    
    engine = ProductSearchEngine()
    engine.load_data("data/sample_products.csv")
    engine.build_index()
    
    queries = [
        "cheap Nike running shoes under 50",
        "premium Trek mountain bike",
        "affordable laptop for students under 500",
        "light bike for trails"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, top_k=2, verbose=True)
        
        if results:
            for result in results:
                print(f"  → {result['product'][:50]}")
                print(f"     {result['brand']} | ${result['price']} | Score: {result['score']:.4f}")
        else:
            print("  No results found")


def example_6_statistics():
    """Example 6: Get dataset statistics."""
    
    print("\n" + "="*80)
    print("Example 6: Dataset Statistics")
    print("="*80)
    
    engine = ProductSearchEngine()
    engine.load_data("data/sample_products.csv")
    engine.build_index()
    
    stats = engine.get_statistics()
    
    print("\nDataset Overview:")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


def example_7_custom_brands():
    """Example 7: Add custom brand recognition."""
    
    print("\n" + "="*80)
    print("Example 7: Custom Brand Recognition")
    print("="*80)
    
    # Add custom brands
    custom_brands = ["MyBrand", "LocalShop", "CustomMaker"]
    
    engine = ProductSearchEngine(custom_brands=custom_brands)
    
    print("\nDefault brands include: Nike, Adidas, Trek, Dell, etc.")
    print(f"Added custom brands: {', '.join(custom_brands)}")
    
    # Test query parsing with custom brand
    from src.nlp_parser import NLPParser
    parser = NLPParser(custom_brands=custom_brands)
    
    test_query = "MyBrand running shoes under 100"
    parsed = parser.parse_query(test_query)
    
    print(f"\nTest query: '{test_query}'")
    print(f"Extracted brands: {parsed['brands']}")
    print(f"Price constraints: {parsed['price_constraints']}")


def example_8_programmatic_api():
    """Example 8: Using the API programmatically."""
    
    print("\n" + "="*80)
    print("Example 8: Programmatic API Usage")
    print("="*80)
    
    # Initialize engine
    engine = ProductSearchEngine()
    engine.load_data("data/sample_products.csv")
    engine.build_index()
    
    # Perform multiple searches programmatically
    queries = ["running", "laptop", "bike"]
    
    print("\nBatch search results:")
    
    all_results = {}
    for query in queries:
        results = engine.search(query, top_k=2)
        all_results[query] = results
        
        print(f"\n'{query}': {len(results)} results")
        if results:
            top_product = results[0]
            print(f"  Top match: {top_product['product'][:40]}")
            print(f"  Score: {top_product['score']:.4f}")
    
    # Export results (example)
    print("\n\nExporting results to dictionary:")
    export = {
        "total_queries": len(queries),
        "results": {
            query: [
                {
                    "product": r["product"],
                    "price": r["price"],
                    "score": r["score"]
                }
                for r in results
            ]
            for query, results in all_results.items()
        }
    }
    
    print(f"Exported {export['total_queries']} queries with results")


def run_all_examples():
    """Run all examples in sequence."""
    
    print("\n")
    print("#" * 80)
    print("# Search Engine Examples - Comprehensive Demonstration")
    print("#" * 80)
    print("\nThis script demonstrates various features of the search engine.")
    print("Press Enter after each example to continue...\n")
    
    examples = [
        ("Basic Search", example_1_basic_search),
        ("Price Filtering", example_2_price_filtering),
        ("Brand Search", example_3_brand_search),
        ("Semantic Search", example_4_semantic_search),
        ("Complex Queries", example_5_complex_queries),
        ("Statistics", example_6_statistics),
        ("Custom Brands", example_7_custom_brands),
        ("Programmatic API", example_8_programmatic_api),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
            input(f"\n[Press Enter to continue to next example...]")
        except Exception as e:
            print(f"\nError in {name}: {e}")
            input(f"\n[Press Enter to continue anyway...]")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]
        example_map = {
            "1": example_1_basic_search,
            "2": example_2_price_filtering,
            "3": example_3_brand_search,
            "4": example_4_semantic_search,
            "5": example_5_complex_queries,
            "6": example_6_statistics,
            "7": example_7_custom_brands,
            "8": example_8_programmatic_api,
        }
        
        if example_num in example_map:
            example_map[example_num]()
        else:
            print(f"Example {example_num} not found.")
            print("Available examples: 1-8")
    else:
        # Run all examples
        run_all_examples()
