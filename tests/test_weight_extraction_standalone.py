"""
Standalone test for weight constraint extraction.
Tests just the NLPParser.extract_numeric_constraints() method.
"""

import re
from typing import Dict


def extract_numeric_constraints(query: str, unit: str = "kg") -> Dict[str, float]:
    """
    Extract numeric constraints for attributes like weight.
    
    Example:
    - "lighter than 12kg" -> {"max": 12}
    - "weight under 15kg" -> {"max": 15}
    
    Args:
        query: User query string
        unit: Unit to look for (default: kg)
        
    Returns:
        Dictionary with min/max constraints
    """
    query_lower = query.lower()
    constraints = {}
    
    under_pattern = f"(?:lighter than|under|below|less than)\\s*(\\d+\\.?\\d*)\\s*{unit}"
    over_pattern = f"(?:heavier than|over|above|more than)\\s*(\\d+\\.?\\d*)\\s*{unit}"
    
    under_match = re.search(under_pattern, query_lower)
    if under_match:
        constraints["max"] = float(under_match.group(1))
    
    over_match = re.search(over_pattern, query_lower)
    if over_match:
        constraints["min"] = float(over_match.group(1))
    
    return constraints


def test_weight_constraints():
    """Test weight constraint extraction."""
    
    print("=" * 80)
    print("WEIGHT CONSTRAINT EXTRACTION - UNIT TEST")
    print("=" * 80)
    
    test_cases = [
        # (query, expected_result)
        ("lighter than 12kg", {"max": 12.0}),
        ("under 15kg", {"max": 15.0}),
        ("below 10kg", {"max": 10.0}),
        ("less than 8.5kg", {"max": 8.5}),
        ("heavier than 20kg", {"min": 20.0}),
        ("over 18kg", {"min": 18.0}),
        ("above 25kg", {"min": 25.0}),
        ("more than 15.5kg", {"min": 15.5}),
        ("bike lighter than 12kg", {"max": 12.0}),
        ("mountain bike under 15kg", {"max": 15.0}),
        ("Trek bike under $500", {}),  # No weight constraint
        ("running shoes", {}),  # No weight constraint
    ]
    
    print("\nTest Results:")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = extract_numeric_constraints(query, unit="kg")
        is_correct = result == expected
        
        if is_correct:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status} | '{query:35s}' | Expected: {str(expected):20s} | Got: {result}")
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 80)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = test_weight_constraints()
    exit(0 if success else 1)
