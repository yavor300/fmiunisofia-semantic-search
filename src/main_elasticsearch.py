"""
Main entry point for Elasticsearch-based product search with data.
"""

import os
import sys

from src.search_engine import ProductSearchEngine


def print_results(results, query):
    """Pretty print search results."""
    print(f"\n{'='*80}")
    print(f"Search Results for: '{query}'")
    print(f"{'='*80}\n")
    
    if not results:
        print("No results found.")
        return
    
    for idx, result in enumerate(results, 1):
        print(f"{idx}. {result['product'][:70]}")
        print(f"   Brand: {result['brand']:<20} Price: ${result['price']}")
        print(f"   Relevance Score: {result['score']:.4f}")
        
        if "description" in result and result["description"]:
            desc = result["description"]
            if len(desc) > 150:
                desc = desc[:150] + "..."
            print(f"   Description: {desc}")
        
        if "category" in result and result["category"]:
            print(f"   Category: {result['category'][:50]}")
        
        if "rating" in result and result["rating"] > 0:
            print(f"   Rating: {result['rating']}/5.0 ({result.get('reviews_count', 0)} reviews)")
        
        if "availability" in result:
            print(f"   Availability: {result['availability']}")
        
        print()


def interactive_search(engine):
    """Run interactive search loop."""
    print("\n" + "="*80)
    print("Interactive Search Mode - Elasticsearch")
    print("="*80)
    print("Enter your search queries (or 'quit' to exit)")
    print("\nExample queries:")
    print("  - 'cheap running shoes'")
    print("  - 'Saucony running shoes'")
    print("  - 'safety vest under 30'")
    print("  - 'solar lights for outdoor'")
    print("  - 'tire pressure gauge'")
    print("-"*80 + "\n")
    
    while True:
        try:
            query = input("Search> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            results = engine.search(query, top_k=5, verbose=True)
            print_results(results, query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def run_demo_queries(engine):
    """Run a set of demonstration queries on data."""
    demo_queries = [
        "running shoes",
        "Saucony running shoes",
        "safety vest under 30",
        "solar post lights",
        "cheap tire gauge",
        "outdoor lights",
        "affordable vest",
        "premium running shoes",
        "budget safety equipment",
        "solar lights for garden"
    ]
    
    print("\n" + "="*80)
    print("Running Demo Queries on Products")
    print("="*80)
    
    for query in demo_queries:
        results = engine.search(query, top_k=3)
        print_results(results, query)
        input("Press Enter to continue...")


def main():
    """Main entry point for Elasticsearch-based search."""
    print("\n" + "="*80)
    print("Domain-Specific Product Search Engine")
    print("Powered by Elasticsearch + Product Data")
    print("="*80)
    print("\nImportant: Make sure Elasticsearch is running!")
    print("Start it with: docker compose up -d")
    print("Check status: docker compose ps")
    print("-"*80)
    
    input("\nPress Enter when Elasticsearch is ready...")
    try:
        print("\nInitializing search engine with Elasticsearch...")
        engine = ProductSearchEngine(use_elasticsearch=True, use_dependency_parsing=True, use_embeddings=True, hybrid_alpha=0.6)
    except Exception as e:
        print(f"\nFailed to connect to Elasticsearch: {e}")
        print("\nMake sure to:")
        print("1. Run: docker compose up -d")
        print("2. Wait ~30 seconds for Elasticsearch to start")
        print("3. Try again")
        return
    amazon_csv = "data/amazon-products.csv"
    if len(sys.argv) > 1:
        amazon_csv = sys.argv[1]
    if not os.path.exists(amazon_csv):
        print(f"\nFile not found: {amazon_csv}")
        print("\nExpected CSV at: data/amazon-products.csv")
        return
    print(f"\nCSV found: {amazon_csv}")
    use_limit = input("Index all products? (Enter limit number or 'all'): ").strip()
    
    limit = None
    if use_limit and use_limit.lower() != 'all':
        try:
            limit = int(use_limit)
            print(f"Will index first {limit} products")
        except ValueError:
            print("Invalid number, indexing all products")
    print(f"\nLoading products...")
    if not engine.load_data(amazon_csv, limit=limit):
        print("Failed to load data. Exiting.")
        return
    print("\nBuilding Elasticsearch index...")
    print("This may take a few minutes for large datasets...")
    
    if not engine.build_index():
        print("Failed to build index. Exiting.")
        return
    stats = engine.get_statistics()
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("\n" + "="*80)
    print("What would you like to do?")
    print("="*80)
    print("1. Run demo queries")
    print("2. Interactive search")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    try:
        if choice == "1":
            run_demo_queries(engine)
        elif choice == "2":
            interactive_search(engine)
        elif choice == "3":
            run_demo_queries(engine)
            interactive_search(engine)
        else:
            print("Invalid choice. Running interactive search by default.")
            interactive_search(engine)
    finally:
        engine.close()
        print("\nCleaned up resources.")


if __name__ == "__main__":
    main()
