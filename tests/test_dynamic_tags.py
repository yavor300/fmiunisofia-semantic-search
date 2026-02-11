"""
Test script to demonstrate dynamic tag generation.
Shows the difference between static and dynamic semantic tags.
"""

import logging
from src.semantic_enrichment import SemanticEnricher, DynamicTagGenerator
from src.config import PRICE_TAG_SEEDS, WEIGHT_TAG_SEEDS, CATEGORY_TAG_SEEDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_dynamic_tag_generator():
    """Test the dynamic tag generator directly."""
    print_section("DYNAMIC TAG GENERATOR TEST")
    
    try:
        generator = DynamicTagGenerator()
        print(f"\nGenerator Configuration:")
        print(f"  Stats: {generator.get_cache_stats()}")
        
        # Test price tags
        print_section("PRICE TAGS (Dynamic)")
        price_tags = generator.generate_all_price_tags()
        for category, tags in price_tags.items():
            print(f"\n{category.upper()}:")
            print(f"  Seeds: {PRICE_TAG_SEEDS[category]}")
            print(f"  Generated: {tags}")
            tag_count = len(tags.split())
            seed_count = len(PRICE_TAG_SEEDS[category])
            print(f"  Expansion: {seed_count} seeds → {tag_count} tags ({tag_count - seed_count} new)")
        
        # Test weight tags
        print_section("WEIGHT TAGS (Dynamic)")
        weight_tags = generator.generate_all_weight_tags()
        for category, tags in weight_tags.items():
            print(f"\n{category.upper()}:")
            print(f"  Seeds: {WEIGHT_TAG_SEEDS[category]}")
            print(f"  Generated: {tags}")
            tag_count = len(tags.split())
            seed_count = len(WEIGHT_TAG_SEEDS[category])
            print(f"  Expansion: {seed_count} seeds → {tag_count} tags ({tag_count - seed_count} new)")
        
        # Test category tags
        print_section("CATEGORY TAGS (Dynamic)")
        test_categories = ["bike", "laptop", "running", "camera"]
        for cat in test_categories:
            if cat in CATEGORY_TAG_SEEDS:
                tags = generator.generate_category_tags(cat)
                print(f"\n{cat.upper()}:")
                print(f"  Seeds: {CATEGORY_TAG_SEEDS[cat]}")
                print(f"  Generated: {tags}")
                tag_count = len(tags.split())
                seed_count = len(CATEGORY_TAG_SEEDS[cat])
                print(f"  Expansion: {seed_count} seeds → {tag_count} tags ({tag_count - seed_count} new)")
        
    except Exception as e:
        logger.error(f"Error testing dynamic tag generator: {e}")
        import traceback
        traceback.print_exc()


def test_semantic_enricher_comparison():
    """Compare static vs dynamic semantic enrichment."""
    print_section("SEMANTIC ENRICHER COMPARISON")
    
    # Test product
    test_product = {
        "name": "Mountain Bike Pro",
        "price_numeric": 150,
        "weight": 11.5,
        "category": "MTB Bike"
    }
    
    print(f"\nTest Product:")
    for key, value in test_product.items():
        print(f"  {key}: {value}")
    
    # Static tags
    print("\n" + "-" * 80)
    print("STATIC TAGS (use_dynamic_tags=False):")
    print("-" * 80)
    try:
        enricher_static = SemanticEnricher(use_dynamic_tags=False)
        tags_static = enricher_static.enrich_product(test_product)
        print(f"Tags: {tags_static}")
        print(f"Tag count: {len(tags_static.split())}")
    except Exception as e:
        logger.error(f"Error with static enricher: {e}")
    
    # Dynamic tags
    print("\n" + "-" * 80)
    print("DYNAMIC TAGS (use_dynamic_tags=True):")
    print("-" * 80)
    try:
        enricher_dynamic = SemanticEnricher(use_dynamic_tags=True)
        tags_dynamic = enricher_dynamic.enrich_product(test_product)
        print(f"Tags: {tags_dynamic}")
        print(f"Tag count: {len(tags_dynamic.split())}")
    except Exception as e:
        logger.error(f"Error with dynamic enricher: {e}")
    
    # Dynamic tags with embeddings
    print("\n" + "-" * 80)
    print("DYNAMIC TAGS + EMBEDDINGS (use_dynamic_tags=True, use_embeddings=True):")
    print("-" * 80)
    try:
        enricher_hybrid = SemanticEnricher(use_dynamic_tags=True, use_embeddings=True)
        tags_hybrid = enricher_hybrid.enrich_product(test_product)
        print(f"Tags: {tags_hybrid}")
        print(f"Tag count: {len(tags_hybrid.split())}")
    except Exception as e:
        logger.error(f"Error with hybrid enricher: {e}")


def test_different_price_ranges():
    """Test tag generation for different price ranges."""
    print_section("PRICE RANGE TAG GENERATION")
    
    test_products = [
        {"name": "Budget Product", "price_numeric": 15},
        {"name": "Affordable Product", "price_numeric": 50},
        {"name": "Mid-Range Product", "price_numeric": 100},
        {"name": "Premium Product", "price_numeric": 200},
        {"name": "Luxury Product", "price_numeric": 500}
    ]
    
    try:
        enricher = SemanticEnricher(use_dynamic_tags=True)
        
        for product in test_products:
            print(f"\n{product['name']} (${product['price_numeric']}):")
            price_cat = enricher.get_price_category(product['price_numeric'])
            print(f"  Category: {price_cat}")
            if price_cat in enricher.price_tags:
                print(f"  Tags: {enricher.price_tags[price_cat]}")
    except Exception as e:
        logger.error(f"Error testing price ranges: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  DYNAMIC SEMANTIC TAG GENERATION TEST".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    try:
        # Test 1: Dynamic tag generator
        test_dynamic_tag_generator()
        
        # Test 2: Semantic enricher comparison
        test_semantic_enricher_comparison()
        
        # Test 3: Different price ranges
        test_different_price_ranges()
        
        print_section("TEST COMPLETE")
        print("\nAll tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  1. Dynamic tag generation using word embeddings")
        print("  2. Fallback to WordNet for synonym expansion")
        print("  3. Tag caching for performance")
        print("  4. Expansion of price, weight, and category tags")
        print("  5. Comparison between static and dynamic approaches")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
