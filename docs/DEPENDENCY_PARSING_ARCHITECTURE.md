# Document-Side Dependency Parsing Architecture

## Problem Statement

Previously, the search engine had an **asymmetric vocabulary problem**:

- **Query-side:** Used dependency parsing to enhance queries with compound terms and relationships
  - Example: "bike with child seat" → "bike child seat bike_with_child_seat"
  
- **Document-side:** Only basic preprocessing (tokenization, stemming)
  - Example: "Mountain Bike with Child Seat" → "mountain bike child seat"

**The Issue:** The query had rich dependency tokens like "bike_with_child_seat", but documents didn't have these same tokens. This created a vocabulary mismatch where the enhanced query terms couldn't match against the document terms.

## Solution: Symmetric Dependency Expansion

The solution is to apply dependency parsing to **both sides** of the search pipeline:

### 1. Document Indexing Pipeline

```
Product Document
    ↓
Extract text (name + description)
    ↓
Dependency Parser → Extract Features:
    - Compound terms: "mountain bike", "carbon fiber"
    - Relationships: "bike_with_child_seat"
    - Key concepts: "bike", "child seat"
    ↓
Add to searchable_content field
    ↓
Index in TF-IDF / Elasticsearch
```

### 2. Query Processing Pipeline (EXISTING)

```
User Query: "bike with child seat"
    ↓
Dependency Parser → Enhance Query:
    "bike child seat bike_with_child_seat"
    ↓
Search against indexed documents
```

### 3. Result: Shared Vocabulary Space

Now both query and documents have the same dependency tokens:

| Component | Tokens |
|-----------|--------|
| **Query** | "bike", "child", "seat", "bike_with_child_seat" |
| **Document** | "mountain", "bike", "child", "seat", "mountain_bike", "bike_with_child_seat" |
| **Match** |  "bike", "child", "seat", "bike_with_child_seat" |

## Implementation Details

### 1. New Method: `TextPreprocessor.extract_dependency_features()`

```python
def extract_dependency_features(self, text: str) -> str:
    """
    Extract dependency parsing features for document indexing.
    
    Returns:
        Space-separated string of additional dependency tokens
    """
    analysis = self.dependency_parser.analyze_query_structure(text)
    
    features = []
    features.extend(analysis.get("compound_terms", []))
    
    for rel in analysis.get("relationships", []):
        features.append(rel["relation"])  # e.g., "bike_with_child_seat"
        features.append(f"{rel['head']} {rel['object']}")  # e.g., "bike child seat"
    
    features.extend(analysis.get("key_concepts", []))
    
    return " ".join(features)
```

### 2. Updated: `SemanticEnricher.enrich_dataframe()`

Now extracts dependency features from each document:

```python
def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    for _, row in df.iterrows():
        # Extract semantic tags (existing)
        tags = self.enrich_product(row.to_dict())
        
        # NEW: Extract dependency features
        dep_features = self._extract_document_dependency_features(row)
        
    df["semantic_tags"] = semantic_tags
    df["dependency_features"] = dependency_features  # NEW
    return df
```

### 3. Updated: `SemanticEnricher.create_searchable_content()`

Includes dependency features in the searchable content:

```python
def create_searchable_content(self, df: pd.DataFrame) -> pd.DataFrame:
    # NEW: Include dependency_features in default fields
    fields = ["name", "brand", "desc", "semantic_tags", "dependency_features"]
    
    for _, row in df.iterrows():
        parts = [str(row[field]) for field in fields if field in row and row[field]]
        searchable_content.append(" ".join(parts))
    
    return df
```

### 4. Updated: `ProductSearchEngine.__init__()`

Passes preprocessor to the enricher:

```python
def __init__(self, ...):
    self.preprocessor = TextPreprocessor(use_dependency_parsing=use_dependency_parsing)
    
    # NEW: Pass preprocessor so enricher can extract dependency features
    self.enricher = SemanticEnricher(
        use_embeddings=use_embeddings,
        preprocessor=self.preprocessor  # NEW
    )
```

## Example Workflow

### Input Document
```python
{
    "name": "Mountain Bike with Child Seat",
    "desc": "Perfect bike for hills and trails",
    "price": 450
}
```

### Step 1: Dependency Feature Extraction
```
Input: "Mountain Bike with Child Seat Perfect bike for hills and trails"

Extracted Features:
- Compound terms: ["mountain bike", "child seat", "perfect bike"]
- Relationships: [
    {"head": "bike", "prep": "with", "object": "child seat", 
     "relation": "bike_with_child_seat"},
    {"head": "bike", "prep": "for", "object": "hills", 
     "relation": "bike_for_hills"}
  ]
- Key concepts: ["mountain bike", "child seat", "bike", "hills", "trails"]

Output: "mountain bike child seat perfect bike bike_with_child_seat bike for hills bike_for_hills trails"
```

### Step 2: Create Searchable Content
```
Searchable Content = name + brand + desc + semantic_tags + dependency_features

Result: "Mountain Bike with Child Seat Trek Perfect bike for hills and trails 
         affordable mountain bike child seat perfect bike bike_with_child_seat 
         bike for hills bike_for_hills trails"
```

### Step 3: Query Matching
```
User Query: "bike with child seat"

Query Enhancement: "bike child seat bike_with_child_seat"

Matching Tokens:
- "bike" 
- "child" 
- "seat" 
- "bike_with_child_seat"  (NEW - this is the key improvement!)

Result: Strong match with high relevance score
```

## Benefits

### 1. Better Vocabulary Matching
- Query and documents now share the same dependency-enhanced vocabulary
- Reduces vocabulary mismatch problems

### 2. Improved Relevance
- Documents with actual prepositional relationships match better
- Example: "bike with child seat" ranks higher than just "bike" or "child seat" separately

### 3. Compound Term Recognition
- Multi-word expressions are preserved
- Example: "carbon fiber frame" is indexed as a single concept, not just "carbon", "fiber", "frame"

### 4. Semantic Relationships
- Prepositional relationships are captured
- Example: "shoes for running" vs "running to buy shoes"

## Performance Considerations

### Indexing Time
- **Impact:** Adds dependency parsing to document indexing (one-time cost)
- **Trade-off:** Slower indexing, but only done once
- **Mitigation:** Description text is limited to first 500 characters

### Index Size
- **Impact:** Adds additional tokens to each document
- **Typical increase:** 10-30% more tokens per document
- **Trade-off:** Larger index, but better matching

### Query Time
- **Impact:** None - query processing already had dependency parsing
- **Benefit:** Better matches improve result quality

## Configuration

Dependency parsing is controlled by the `use_dependency_parsing` flag:

```python
# Enable dependency parsing for both documents and queries
search_engine = ProductSearchEngine(
    use_dependency_parsing=True  # Default: False
)
```

When enabled:
- Documents: Dependency features extracted during indexing
- Queries: Dependency enhancements applied during search

When disabled:
- Both sides fall back to basic preprocessing
- Faster but less semantic understanding

## Testing

Run the test script to see dependency parsing in action:

```bash
python test_dependency_indexing.py
```

This demonstrates:
1. Dependency feature extraction from documents
2. Full enrichment pipeline with dependency features
3. Query enhancement for comparison
4. Vocabulary matching demonstration

## Conclusion

By applying dependency parsing to both documents and queries, we create a **symmetric vocabulary space** where enhanced query tokens can match against enhanced document tokens. This improves retrieval quality for complex queries with grammatical relationships, compound terms, and prepositional phrases.

The key insight: **What you do to the query, you should also do to the documents.**
