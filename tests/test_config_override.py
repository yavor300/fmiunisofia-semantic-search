"""
Example test script demonstrating how to override configuration parameters.
This shows how test files can customize behavior without modifying config.py.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_override_word_embeddings():
    """Test overriding word embedding configuration."""
    from src.word_embeddings import SemanticEmbeddings
    from src.config import build_semantic_mappings
    
    print("\n" + "="*80)
    print("TEST: Word Embeddings Configuration Override")
    print("="*80)
    
    # Custom semantic mappings for testing
    custom_mappings = {
        "test": ["example", "demo", "trial"],
        "fast": ["quick", "speedy", "rapid"]
    }
    
    # Create embeddings with custom configuration
    embeddings = SemanticEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Override model
        use_cache=False,                 # Disable cache for testing
        semantic_mappings=custom_mappings # Custom mappings
    )
    
    print(f"\nEmbeddings initialized with custom config:")
    print(f"  Model: {embeddings.model_name}")
    print(f"  Cache enabled: {embeddings.use_cache}")
    print(f"  Custom mappings: {list(custom_mappings.keys())}")
    
    # Test query expansion with custom mappings
    expansions = embeddings.expand_query("test fast")
    print(f"\nQuery expansion for 'test fast':")
    print(f"  Expanded terms: {expansions}")


def test_override_semantic_enrichment():
    """Test overriding semantic enrichment configuration."""
    from src.semantic_enrichment import SemanticEnricher
    
    print("\n" + "="*80)
    print("TEST: Semantic Enrichment Configuration Override")
    print("="*80)
    
    # Custom price thresholds for testing
    custom_price_thresholds = {
        "budget": (0, 50),
        "mid_range": (50, 150),
        "premium": (150, float('inf'))
    }
    
    # Custom weight thresholds for testing
    custom_weight_thresholds = {
        "light": (0, 10),
        "heavy": (10, float('inf'))
    }
    
    # Create enricher with custom configuration
    enricher = SemanticEnricher(
        use_embeddings=False,            # Disable embeddings for faster testing
        use_dynamic_tags=False,          # Use static tags for predictable testing
        price_thresholds=custom_price_thresholds,
        weight_thresholds=custom_weight_thresholds
    )
    
    print(f"\nEnricher initialized with custom config:")
    print(f"  Use embeddings: {enricher.use_embeddings}")
    print(f"  Use dynamic tags: {enricher.use_dynamic_tags}")
    print(f"  Price categories: {list(custom_price_thresholds.keys())}")
    print(f"  Weight categories: {list(custom_weight_thresholds.keys())}")
    
    # Test with custom thresholds
    test_product = {"price_numeric": 75, "weight": 8}
    price_cat = enricher.get_price_category(test_product["price_numeric"])
    weight_cat = enricher.get_weight_category(test_product["weight"])
    
    print(f"\nTest product categorization:")
    print(f"  Product: ${test_product['price_numeric']}, {test_product['weight']}kg")
    print(f"  Price category: {price_cat}")
    print(f"  Weight category: {weight_cat}")


def test_override_dynamic_tag_generator():
    """Test overriding dynamic tag generator configuration."""
    from src.semantic_enrichment import DynamicTagGenerator
    
    print("\n" + "="*80)
    print("TEST: Dynamic Tag Generator Configuration Override")
    print("="*80)
    
    # Create generator with custom configuration for testing
    generator = DynamicTagGenerator(
        use_embeddings=False,   # Disable embeddings for faster testing
        use_wordnet=True,       # Only use WordNet
        cache_tags=False,       # Disable cache for testing
        max_synonyms=2          # Limit synonyms
    )
    
    print(f"\nTag generator initialized with custom config:")
    print(f"  Stats: {generator.get_cache_stats()}")
    
    # Test tag generation
    tags = generator.generate_tags(
        category="test",
        seed_words=["cheap", "budget"],
        max_per_seed=2
    )
    
    print(f"\nGenerated tags:")
    print(f"  Seeds: ['cheap', 'budget']")
    print(f"  Tags: {tags}")


def test_override_text_preprocessor():
    """Test overriding text preprocessor configuration."""
    from src.preprocessing import TextPreprocessor
    
    print("\n" + "="*80)
    print("TEST: Text Preprocessor Configuration Override")
    print("="*80)
    
    # Create preprocessor with custom configuration
    preprocessor = TextPreprocessor(
        language="english",              # Specify language
        use_dependency_parsing=False     # Disable for faster testing
    )
    
    print(f"\nPreprocessor initialized with custom config:")
    print(f"  Language: english")
    print(f"  Dependency parsing: False")
    
    # Test preprocessing
    test_text = "Running shoes for the best athletes!"
    processed = preprocessor.preprocess(test_text)
    
    print(f"\nText preprocessing:")
    print(f"  Input: '{test_text}'")
    print(f"  Output: '{processed}'")


def test_override_nlp_parser():
    """Test overriding NLP parser configuration."""
    from src.nlp_parser import NLPParser
    
    print("\n" + "="*80)
    print("TEST: NLP Parser Configuration Override")
    print("="*80)
    
    # Create parser with custom brands for testing
    custom_test_brands = ["testbrand", "demobrand"]
    
    parser = NLPParser(
        custom_brands=custom_test_brands
    )
    
    print(f"\nParser initialized with custom brands:")
    print(f"  Custom brands: {custom_test_brands}")
    print(f"  Total brands: {len(parser.brands)}")
    
    # Test brand extraction
    test_query = "TestBrand shoes under 100"
    parsed = parser.parse_query(test_query)
    
    print(f"\nQuery parsing:")
    print(f"  Query: '{test_query}'")
    print(f"  Extracted brands: {parsed['brands']}")
    print(f"  Price constraints: {parsed['price_constraints']}")


def test_override_search_indexer():
    """Test overriding search indexer configuration."""
    from src.indexer import SearchIndexer
    
    print("\n" + "="*80)
    print("TEST: Search Indexer Configuration Override")
    print("="*80)
    
    # Create indexer with custom configuration for testing
    indexer = SearchIndexer(
        max_features=1000,      # Smaller feature set for testing
        ngram_range=(1, 1),     # Only unigrams for testing
        min_df=1,               # Keep all terms
        sublinear_tf=False      # Disable for testing
    )
    
    print(f"\nIndexer initialized with custom config:")
    print(f"  Max features: 1000")
    print(f"  N-gram range: (1, 1) - unigrams only")
    print(f"  Sublinear TF: False")


def test_override_search_engine():
    """Test overriding search engine configuration."""
    from src.search_engine import ProductSearchEngine
    
    print("\n" + "="*80)
    print("TEST: Search Engine Configuration Override")
    print("="*80)
    
    # Create search engine with custom configuration for testing
    engine = ProductSearchEngine(
        custom_brands=["testbrand"],
        use_elasticsearch=False,         # Use local indexer for testing
        use_dependency_parsing=False,    # Disable for faster testing
        use_embeddings=False,            # Disable for faster testing
        hybrid_alpha=0.7                 # Custom hybrid weight
    )
    
    print(f"\nSearch engine initialized with custom config:")
    print(f"  Elasticsearch: {engine.use_elasticsearch}")
    print(f"  Dependency parsing: {engine.use_dependency_parsing}")
    print(f"  Word embeddings: {engine.use_embeddings}")
    print(f"  Hybrid alpha: {engine.hybrid_alpha}")


def main():
    """Run all override tests."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  CONFIGURATION OVERRIDE TEST EXAMPLES".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    print("\nThis script demonstrates how to override default configuration")
    print("parameters when creating instances for testing purposes.")
    print("\nAll classes now accept optional parameters that default to config.py")
    print("values but can be overridden as needed for testing.\n")
    
    try:
        # Test each component
        test_override_word_embeddings()
        test_override_semantic_enrichment()
        test_override_dynamic_tag_generator()
        test_override_text_preprocessor()
        test_override_nlp_parser()
        test_override_search_indexer()
        test_override_search_engine()
        
        print("\n" + "="*80)
        print("ALL CONFIGURATION OVERRIDE TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
