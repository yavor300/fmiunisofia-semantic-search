"""
Test script to demonstrate document-side dependency parsing.

This shows how dependency features are now extracted from both documents
and queries, ensuring vocabulary matching.
"""

import pandas as pd
from src.preprocessing import TextPreprocessor
from src.semantic_enrichment import SemanticEnricher


def test_dependency_feature_extraction():
    """Test that dependency features are extracted from documents."""
    
    print("=" * 80)
    print("DEPENDENCY PARSING - DOCUMENT SIDE")
    print("=" * 80)
    
    # Initialize preprocessor with dependency parsing enabled
    preprocessor = TextPreprocessor(use_dependency_parsing=True)
    
    # Test documents
    test_documents = [
        {
            "name": "Mountain Bike with Child Seat",
            "brand": "Trek",
            "desc": "Perfect bike for hills and trails",
            "price_numeric": 450
        },
        {
            "name": "Carbon Fiber Frame Road Bike",
            "brand": "Specialized",
            "desc": "Lightweight bike for racing",
            "price_numeric": 1200
        },
        {
            "name": "Running Shoes for Marathon Training",
            "brand": "Nike",
            "desc": "Comfortable shoes with excellent cushioning",
            "price_numeric": 120
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(test_documents)
    
    # Initialize enricher with preprocessor
    enricher = SemanticEnricher(
        use_embeddings=False,
        use_dynamic_tags=False,
        preprocessor=preprocessor
    )
    
    print("\n1. Testing dependency feature extraction from documents:")
    print("-" * 80)
    
    for idx, row in df.iterrows():
        doc_text = f"{row['name']} {row['desc']}"
        features = preprocessor.extract_dependency_features(doc_text)
        
        print(f"\nDocument {idx + 1}: {row['name']}")
        print(f"Description: {row['desc']}")
        print(f"Extracted Features: {features}")
    
    # Test enrichment pipeline
    print("\n\n2. Testing full enrichment pipeline:")
    print("-" * 80)
    
    df_enriched = enricher.enrich_dataframe(df)
    df_enriched = enricher.create_searchable_content(df_enriched)
    
    for idx, row in df_enriched.iterrows():
        print(f"\nDocument {idx + 1}: {row['name']}")
        print(f"Semantic Tags: {row['semantic_tags']}")
        print(f"Dependency Features: {row['dependency_features']}")
        print(f"Searchable Content Preview: {row['searchable_content']}")
    
    # Test query enhancement for comparison
    print("\n\n3. Query-side dependency parsing (for comparison):")
    print("-" * 80)
    
    test_queries = [
        "bike with child seat",
        "carbon fiber frame",
        "shoes for running"
    ]
    
    for query in test_queries:
        enhanced = preprocessor.enhance_query(query)
        print(f"\nOriginal Query: {query}")
        print(f"Enhanced Query: {enhanced}")
    
    print("\n" + "=" * 80)
    print("VOCABULARY MATCHING DEMONSTRATION")
    print("=" * 80)
    print("\nNOW, when a user searches for 'bike with child seat':")
    print("  - Query is enhanced to: 'bike child seat bike_with_child_seat'")
    print("  - Document 1 has dependency features: 'mountain bike bike with child seat'")
    print("  - Both share tokens like 'bike', 'child', 'seat', 'mountain bike'")
    print("  - Result: Better matching and ranking!")
    print("=" * 80)


if __name__ == "__main__":
    test_dependency_feature_extraction()
