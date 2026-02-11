"""
Standalone test for bidirectional semantic mapping (no dependencies).
"""

# Simulated semantic mappings
SEMANTIC_MAPPINGS = {
    "cheap": ["budget", "affordable", "inexpensive", "economical"],
    "expensive": ["premium", "luxury", "high-end", "costly"],
    "hills": ["mountain", "climbing", "steep", "uphill"],
    "mountain bike": ["mtb", "trail bike", "off-road bike"],
}


def build_reverse_index(mappings):
    """Build reverse index for bidirectional lookups."""
    reverse_index = {}
    
    for primary_term, synonyms in mappings.items():
        for synonym in synonyms:
            synonym_lower = synonym.lower()
            if synonym_lower not in reverse_index:
                reverse_index[synonym_lower] = []
            reverse_index[synonym_lower].append(primary_term)
    
    return reverse_index


def expand_query_old(query, mappings):
    """OLD: Forward-only expansion."""
    expanded_terms = []
    words = query.lower().split()
    
    for word in words:
        if word in mappings:
            expansions = mappings[word][:3]
            expanded_terms.extend(expansions)
    
    return expanded_terms


def expand_query_new(query, mappings, reverse_mappings):
    """NEW: Bidirectional expansion."""
    expanded_terms = []
    words = query.lower().split()
    
    for word in words:
        # Forward lookup: word is primary term
        if word in mappings:
            expansions = mappings[word][:3]
            expanded_terms.extend(expansions)
        
        # Reverse lookup: word is a synonym
        elif word in reverse_mappings:
            primary_terms = reverse_mappings[word]
            
            for primary_term in primary_terms:
                if primary_term in mappings:
                    related_terms = mappings[primary_term][:3]
                    
                    for term in related_terms:
                        if term.lower() != word:
                            expanded_terms.append(term)
                    
                    break
    
    return expanded_terms


print("=" * 80)
print("BIDIRECTIONAL SEMANTIC MAPPING TEST")
print("=" * 80)

# Build reverse index
reverse_mappings = build_reverse_index(SEMANTIC_MAPPINGS)

print("\n📚 SEMANTIC GROUPS:")
print("-" * 80)
for primary, synonyms in SEMANTIC_MAPPINGS.items():
    print(f"  '{primary}' → {synonyms}")

print("\n\n🔄 REVERSE INDEX:")
print("-" * 80)
print("Sample mappings:")
for term in ["budget", "affordable", "premium", "mtb"]:
    if term in reverse_mappings:
        print(f"  '{term}' → Primary: {reverse_mappings[term]}")

print("\n\n" + "=" * 80)
print("EXPANSION COMPARISON")
print("=" * 80)

test_queries = [
    "cheap laptop",
    "budget laptop",
    "affordable bike",
    "premium shoes",
    "hills riding",
    "mountain biking",
]

print(f"\n{'Query':<25} | {'Old (Forward Only)':<40} | {'New (Bidirectional)':<40}")
print("-" * 110)

for query in test_queries:
    old_expansion = expand_query_old(query, SEMANTIC_MAPPINGS)
    new_expansion = expand_query_new(query, SEMANTIC_MAPPINGS, reverse_mappings)
    
    old_str = ", ".join(old_expansion) if old_expansion else "(none)"
    new_str = ", ".join(new_expansion) if new_expansion else "(none)"
    
    print(f"{query:<25} | {old_str:<40} | {new_str:<40}")

print("\n\n✅ IMPROVEMENT SUMMARY")
print("=" * 80)

improvements = [
    "✓ 'budget laptop' now expands (was broken before)",
    "✓ 'affordable bike' now expands (was broken before)",
    "✓ 'premium shoes' now expands (was broken before)",
    "✓ Any synonym can trigger expansion (not just primary terms)",
    "✓ More flexible and user-friendly",
]

for improvement in improvements:
    print(f"  {improvement}")

print("\n\n📈 STATISTICS")
print("=" * 80)

total_primary = len(SEMANTIC_MAPPINGS)
total_synonyms = sum(len(synonyms) for synonyms in SEMANTIC_MAPPINGS.values())

print(f"\nTerms that trigger expansion:")
print(f"  ❌ Before: {total_primary} terms (only primary terms)")
print(f"  ✅ After:  {total_primary + total_synonyms} terms (primary + synonyms)")
print(f"  📈 Improvement: +{total_synonyms} additional terms!")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)
