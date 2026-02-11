"""
Quick diagnostic script to check if description is in results.
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_description_flow():
    """Test the description data flow."""
    
    print("="*70)
    print("DESCRIPTION FIELD DIAGNOSTIC TEST")
    print("="*70)
    
    # Create a simple test DataFrame
    test_df = pd.DataFrame({
        'product': ['Test Product 1', 'Test Product 2'],
        'brand': ['TestBrand', 'TestBrand2'],
        'price': [99.99, 49.99],
        'desc': ['This is test description 1', 'This is test description 2'],
        'similarity_score': [0.9, 0.8]
    })
    
    print("\n1. Test DataFrame created:")
    print(f"   Columns: {list(test_df.columns)}")
    print(f"   'desc' in columns: {'desc' in test_df.columns}")
    print(f"   First desc value: {test_df.iloc[0]['desc']}")
    
    # Simulate the code in search_engine.py
    print("\n2. Testing the formatting logic:")
    results = []
    for _, row in test_df.iterrows():
        result = {
            "product": row["product"],
            "brand": row["brand"],
            "price": row["price"],
            "score": row["similarity_score"]
        }
        
        # Test OLD method (buggy)
        if "desc" in row:  # This checks index, not columns!
            print(f"   OLD method: 'desc' in row = True (WRONG!)")
        else:
            print(f"   OLD method: 'desc' in row = False (BUG!)")
        
        # Test NEW method (correct)
        if "desc" in test_df.columns and pd.notna(row.get("desc")):
            result["description"] = str(row["desc"])
            print(f"   NEW method: Added description = '{result['description'][:30]}...'")
        
        results.append(result)
    
    print("\n3. Final results:")
    for i, r in enumerate(results, 1):
        print(f"\n   Result {i}:")
        for key, value in r.items():
            if key == "description":
                print(f"      {key}: {value}")
            else:
                print(f"        {key}: {value}")
    
    # Check if description is present
    has_description = "description" in results[0]
    
    print("\n" + "="*70)
    if has_description:
        print("SUCCESS: Description field is present in results!")
    else:
        print("FAIL: Description field is missing from results!")
    print("="*70)
    
    return has_description


if __name__ == '__main__':
    success = test_description_flow()
    sys.exit(0 if success else 1)
