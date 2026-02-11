"""
Test script to verify the embeddings fix for handling empty/None documents.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.word_embeddings import SemanticEmbeddings

def test_embeddings_with_problematic_data():
    """Test embeddings with various edge cases."""
    
    print("="*80)
    print("Testing Embeddings with Edge Cases")
    print("="*80)
    
    # Initialize embeddings
    print("\n[1] Initializing embeddings...")
    try:
        embeddings = SemanticEmbeddings()
        print("Embeddings initialized")
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return False
    
    # Test 1: Normal documents
    print("\n[2] Test with normal documents...")
    try:
        query = "budget safety equipment"
        docs = [
            "Budget safety gear for construction",
            "Affordable protective equipment",
            "Premium safety tools"
        ]
        
        results = embeddings.compute_similarity(query, docs)
        print(f"Normal documents work: {len(results)} results")
        for idx, score in results[:2]:
            print(f"  - Doc {idx}: {score:.3f}")
    except Exception as e:
        print(f"Failed: {e}")
        return False
    
    # Test 2: Documents with None values
    print("\n[3] Test with None values...")
    try:
        query = "budget equipment"
        docs = [
            "Budget equipment",
            None,
            "Affordable tools",
            None
        ]
        
        results = embeddings.compute_similarity(query, docs)
        print(f"None values handled: {len(results)} results")
        for idx, score in results:
            print(f"  - Doc {idx}: {score:.3f}")
    except Exception as e:
        print(f"Failed: {e}")
        return False
    
    # Test 3: Documents with empty strings
    print("\n[4] Test with empty strings...")
    try:
        query = "safety equipment"
        docs = [
            "Safety equipment",
            "",
            "   ",  # Whitespace only
            "Protective gear",
            ""
        ]
        
        results = embeddings.compute_similarity(query, docs)
        print(f"Empty strings handled: {len(results)} results")
        for idx, score in results:
            print(f"  - Doc {idx}: {score:.3f}")
    except Exception as e:
        print(f"Failed: {e}")
        return False
    
    # Test 4: Mixed problematic data
    print("\n[5] Test with mixed problematic data...")
    try:
        query = "affordable products"
        docs = [
            "Budget product A",
            None,
            "",
            "Affordable product B",
            "   ",
            None,
            "Premium product C"
        ]
        
        results = embeddings.compute_similarity(query, docs, top_k=3)
        print(f"Mixed data handled: {len(results)} results")
        for idx, score in results:
            doc_preview = docs[idx] if docs[idx] else "[empty]"
            print(f"  - Doc {idx}: {score:.3f} - {doc_preview}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: All empty documents
    print("\n[6] Test with all empty documents...")
    try:
        query = "test query"
        docs = [None, "", "   ", None]
        
        results = embeddings.compute_similarity(query, docs)
        print(f"All empty handled: {len(results)} results (expected 0)")
    except Exception as e:
        print(f"Failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("All edge case tests passed!")
    print("="*80)
    return True


if __name__ == "__main__":
    print("\nTesting embeddings fix for edge cases...\n")
    success = test_embeddings_with_problematic_data()
    sys.exit(0 if success else 1)
