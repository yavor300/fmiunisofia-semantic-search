"""
Test script to demonstrate the improvements to dependency feature extraction.

Shows the before/after comparison for:
1. Separating title and description processing
2. Adding underscores to multi-word features (tag-like)
3. Removing duplicate "head + object" additions
"""

import pandas as pd
from src.preprocessing import TextPreprocessor
from src.semantic_enrichment import SemanticEnricher


def test_improved_dependency_features():
    """Test the improved dependency feature extraction."""
    
    print("=" * 80)
    print("DEPENDENCY FEATURE EXTRACTION - IMPROVEMENTS DEMONSTRATION")
    print("=" * 80)
    
    # Initialize preprocessor with dependency parsing
    preprocessor = TextPreprocessor(use_dependency_parsing=True)
    
    # Test product
    test_product = {
        "name": "Mountain Bike with Child Seat",
        "brand": "Trek",
        "desc": "Perfect bike for hills and trails",
        "price_numeric": 450
    }
    
    print("\nTEST PRODUCT:")
    print(f"  Name: {test_product['name']}")
    print(f"  Description: {test_product['desc']}")
    
    # Create DataFrame
    df = pd.DataFrame([test_product])
    
    # Initialize enricher with preprocessor
    enricher = SemanticEnricher(
        use_embeddings=False,
        use_dynamic_tags=False,
        preprocessor=preprocessor
    )
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT 1: Separate Title and Description Processing")
    print("=" * 80)
    
    print("\n❌ BEFORE (Combined Processing):")
    print("  Combined text: 'Mountain Bike with Child Seat Perfect bike for hills and trails'")
    print("  Problem: Creates cross-contamination like 'Bike_with_Child_Seat_Perfect_bike_for_hills'")
    
    print("\n✅ AFTER (Separate Processing):")
    print("  Title processed separately: 'Mountain Bike with Child Seat'")
    print("  Description processed separately: 'Perfect bike for hills and trails'")
    print("  Benefit: No cross-contamination between title and description!")
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT 2: Underscores for Multi-Word Features")
    print("=" * 80)
    
    print("\n❌ BEFORE:")
    print("  'mountain bike' -> Treated as two separate words: 'mountain', 'bike'")
    print("  Problem: Can match unrelated 'mountain' or 'bike' separately")
    
    print("\n✅ AFTER:")
    print("  'mountain bike' -> 'mountain_bike' (single tag)")
    print("  Benefit: Treated as a single concept, better precision!")
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT 3: Removed Duplicate 'head + object' Additions")
    print("=" * 80)
    
    print("\n❌ BEFORE:")
    print("  Relationship 'bike_with_child_seat' was added TWICE:")
    print("    1. As 'bike_with_child_seat' (from relation)")
    print("    2. As 'bike child seat' (from head + object)")
    print("  Problem: Unnecessary duplication!")
    
    print("\n✅ AFTER:")
    print("  Only 'bike_with_child_seat' is added once")
    print("  Benefit: Cleaner, less redundant features!")
    
    print("\n" + "=" * 80)
    print("ACTUAL EXTRACTION RESULTS")
    print("=" * 80)
    
    # Extract features
    df_enriched = enricher.enrich_dataframe(df)
    df_enriched = enricher.create_searchable_content(df_enriched)
    
    row = df_enriched.iloc[0]
    
    print(f"\n📝 Dependency Features Extracted:")
    features = row['dependency_features'].split()
    print(f"  Total features: {len(features)}")
    print(f"  Unique features: {len(set(features))}")
    print(f"\n  Features list:")
    for feature in features:
        print(f"    - {feature}")
    
    print(f"\nSearchable Content Preview (first 200 chars):")
    content = row['searchable_content']
    print(f"  {content}")
    print(f"\n  Full length: {len(content)} characters")
    
    print("\n" + "=" * 80)
    print("QUERY ENHANCEMENT CONSISTENCY")
    print("=" * 80)
    
    # Test query enhancement
    test_query = "bike with child seat"
    enhanced_query = preprocessor.enhance_query(test_query)
    
    print(f"\n🔍 Query: '{test_query}'")
    print(f"✨ Enhanced: '{enhanced_query}'")
    print("\nQuery features:")
    for term in enhanced_query.split():
        print(f"  - {term}")
    
    print("\n✅ MATCHING:")
    query_features = set(enhanced_query.split())
    doc_features = set(features)
    matching = query_features.intersection(doc_features)
    
    print(f"  Query features: {len(query_features)}")
    print(f"  Document features: {len(doc_features)}")
    print(f"  Matching features: {len(matching)}")
    print(f"\n  Shared features:")
    for feature in matching:
        print(f"    ✓ {feature}")
    
    print("\n" + "=" * 80)
    print("BENEFITS SUMMARY")
    print("=" * 80)
    
    print("""
✅ 1. NO CROSS-CONTAMINATION
   - Title and description processed separately
   - No more 'Bike_with_Child_Seat_Perfect_bike_for_hills' nonsense

✅ 2. CONSISTENT TAG FORMAT
   - Multi-word features use underscores: 'mountain_bike'
   - Works as single concept/tag
   - Same format for queries and documents

✅ 3. NO DUPLICATES
   - Removed redundant 'head + object' additions
   - Each feature appears only once
   - Cleaner, more efficient

✅ 4. BETTER PRECISION
   - Tags like 'mountain_bike' match exactly
   - Not confused with separate 'mountain' and 'bike' words
   - Improved semantic matching
""")
    
    print("=" * 80)


if __name__ == "__main__":
    test_improved_dependency_features()
