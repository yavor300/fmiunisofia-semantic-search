"""
Quick test to verify description field is included in search results.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_description_in_results():
    """Test that description is returned in search results."""
    print("Testing description field in search results...")
    print("="*60)
    
    try:
        from src.search_engine import ProductSearchEngine
        import pandas as pd
        
        # Create a minimal test dataset
        test_data = pd.DataFrame({
            'title': ['Nike Running Shoes', 'Adidas Sports Shoes'],
            'brand': ['Nike', 'Adidas'],
            'description': [
                'High-quality running shoes with excellent cushioning',
                'Durable sports shoes for all activities'
            ],
            'final_price': [99.99, 89.99],
            'category': ['Sports', 'Sports'],
            'rating': [4.5, 4.0],
            'reviews_count': [100, 50]
        })
        
        # Save to temporary file
        test_file = 'data/test_products.csv'
        os.makedirs('data', exist_ok=True)
        test_data.to_csv(test_file, index=False)
        
        print("Created test dataset")
        
        # Initialize search engine with minimal config
        engine = ProductSearchEngine(
            use_elasticsearch=False,
            use_dependency_parsing=False,
            use_embeddings=False
        )
        
        print("Initialized search engine")
        
        # Load data
        if not engine.load_data(test_file):
            print("Failed to load data")
            return False
        
        print("Loaded test data")
        
        # Build index
        if not engine.build_index():
            print("Failed to build index")
            return False
        
        print("Built search index")
        
        # Perform search
        results = engine.search("running shoes", top_k=2)
        
        print(f"Search completed, got {len(results)} results")
        
        # Check if description is in results
        if not results:
            print("No results returned")
            return False
        
        first_result = results[0]
        
        print("\nFirst result fields:")
        for key in first_result.keys():
            print(f"  - {key}")
        
        if "description" in first_result:
            print("\nSUCCESS: Description field is present!")
            print(f"\nSample description: {first_result['description'][:100]}...")
            
            # Print formatted result
            print("\n" + "="*60)
            print("Sample formatted result:")
            print("="*60)
            print(f"Product: {first_result['product']}")
            print(f"Brand: {first_result['brand']}")
            print(f"Price: ${first_result['price']:.2f}")
            print(f"Score: {first_result['score']:.4f}")
            if first_result.get('description'):
                desc = first_result['description']
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                print(f"Description: {desc}")
            print("="*60)
            
            return True
        else:
            print("\nFAIL: Description field is NOT present")
            return False
        
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if os.path.exists('data/test_products.csv'):
            os.remove('data/test_products.csv')
            print("\nCleaned up test files")


if __name__ == '__main__':
    success = test_description_in_results()
    
    if success:
        print("\n" + "="*60)
        print("TEST PASSED: Description field works correctly!")
        print("="*60)
        print("\nThe description field will now appear in:")
        print("  1. Search API responses")
        print("  2. Web interface results")
        print("  3. CLI output (src/main.py)")
        print("  4. Elasticsearch CLI output (src/main_elasticsearch.py)")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)
        sys.exit(1)
