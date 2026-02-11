"""
Test script to demonstrate the weight filtering feature.

This shows how weight constraints are now extracted from queries
and applied as filters during search.
"""

from src.nlp_parser import NLPParser


def test_weight_constraint_extraction():
    """Test that weight constraints are correctly extracted from queries."""
    
    print("=" * 80)
    print("WEIGHT FILTERING FEATURE - DEMONSTRATION")
    print("=" * 80)
    
    parser = NLPParser()
    
    # Test queries with weight constraints
    test_queries = [
        "lightweight bike under 12kg",
        "mountain bike lighter than 15kg",
        "bike weighs less than 10kg",
        "heavy bike over 20kg",
        "bike heavier than 18kg for downhill",
        "bike weight under 8kg for racing",
        "Nike shoes under $100",  # No weight constraint
        "affordable mountain bike",  # No weight constraint
    ]
    
    print("\n1. Testing Weight Constraint Extraction:")
    print("-" * 80)
    
    for query in test_queries:
        parsed = parser.parse_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"  Brands: {parsed['brands']}")
        print(f"  Price constraints: {parsed['price_constraints']}")
        print(f"  Weight constraints: {parsed['weight_constraints']}")
        print(f"  Clean query: '{parsed['clean_query']}'")
    
    print("\n\n2. Detailed Weight Pattern Matching:")
    print("-" * 80)
    
    weight_test_cases = [
        ("lighter than 12kg", {"max": 12.0}),
        ("under 15kg", {"max": 15.0}),
        ("below 10kg", {"max": 10.0}),
        ("less than 8.5kg", {"max": 8.5}),
        ("heavier than 20kg", {"min": 20.0}),
        ("over 18kg", {"min": 18.0}),
        ("above 25kg", {"min": 25.0}),
        ("more than 15.5kg", {"min": 15.5}),
    ]
    
    all_passed = True
    for query, expected in weight_test_cases:
        result = parser.extract_numeric_constraints(query, unit="kg")
        passed = result == expected
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{status} | Query: '{query:25s}' | Expected: {expected} | Got: {result}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    print("INTEGRATION WITH SEARCH PIPELINE")
    print("=" * 80)
    print("\nHow it works:")
    print("1. User query: 'Nike running shoes under 12kg'")
    print("2. NLPParser extracts:")
    print("   - Brands: ['nike']")
    print("   - Weight: {'max': 12}")
    print("   - Clean query: 'running shoes'")
    print("3. SearchEngine builds filters:")
    print("   - filters['brand'] = ['nike']")
    print("   - filters['weight_max'] = 12")
    print("4. Indexer applies filters:")
    print("   - Filters to products where brand='nike' AND weight <= 12")
    print("5. Returns: Only Nike running shoes under 12kg")
    
    print("\n" + "=" * 80)
    print("EXAMPLE QUERIES THAT NOW WORK")
    print("=" * 80)
    
    example_queries = [
        "lightweight carbon bike under 8kg",
        "mountain bike lighter than 15kg under $500",
        "road bike weighs less than 10kg",
        "Trek bike under 12kg for racing",
        "heavy downhill bike over 20kg",
    ]
    
    print("\nThe following queries now support weight filtering:\n")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. \"{query}\"")
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)


if __name__ == "__main__":
    test_weight_constraint_extraction()
