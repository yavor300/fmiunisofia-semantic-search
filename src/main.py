"""
Main entry point for the Domain-Specific Intelligent Product Search Engine.

This module provides a command-line interface for testing the search engine
with either sample data or real product data.
"""

import os
import sys

from src.search_engine import ProductSearchEngine
from src.sample_data import generate_sample_data

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
        
        if "category" in result:
            print(f"   Category: {result['category']}")
        
        if "top_terms" in result:
            terms = ", ".join([f"{term}({weight})" for term, weight in result["top_terms"][:3]])
            print(f"   Top Terms: {terms}")
        
        print()


def interactive_search(engine):
    """Run interactive search loop."""
    print("\n" + "="*80)
    print("Interactive Search Mode")
    print("="*80)
    print("Enter your search queries (or 'quit' to exit)")
    print("Example queries:")
    print("  - 'cheap running shoes'")
    print("  - 'budget bike under 300'")
    print("  - 'premium laptop Dell'")
    print("  - 'light mountain bike for trails'")
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


def run_demo_queries(engine):
    """Run a set of demonstration queries."""
    demo_queries = [
        "cheap running shoes",
        "budget bike under 300",
        "premium laptop",
        "light mountain bike",
        "Nike shoes for running",
        "expensive luxury bike",
        "affordable laptop for students",
        "bike with child seat",
        "Trek mountain bike",
        "headphones with noise cancelling"
    ]
    
    print("\n" + "="*80)
    print("Running Demo Queries")
    print("="*80)
    
    for query in demo_queries:
        results = engine.search(query, top_k=3)
        print_results(results, query)
        input("Press Enter to continue...")


def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("Domain-Specific Intelligent Product Search Engine")
    print("="*80)

    engine = ProductSearchEngine()

    sample_data_path = "data/sample_products.csv"

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        if not os.path.exists(data_path):
            print(f"ERROR: File not found: {data_path}")
            return
    else:

        if not os.path.exists(sample_data_path):
            print("\nGenerating sample data...")
            generate_sample_data(sample_data_path)
        data_path = sample_data_path

    print(f"\nLoading data from: {data_path}")
    if not engine.load_data(data_path):
        print("Failed to load data. Exiting.")
        return

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


if __name__ == "__main__":
    main()
