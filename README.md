# Domain-Specific Intelligent Product Search Engine

An advanced Information Retrieval (IR) system designed for domain-specific e-commerce environments. This system utilizes Natural Language Processing (NLP) and Vector Space Models to understand user intent and product semantics, going beyond simple keyword matching.

## Features

### Hybrid Search Mechanism
- **Inverted Indexing:** Fast document retrieval using TF-IDF
- **Vector Space Model:** Semantic matching using cosine similarity
- **Parametric Filtering:** Hard filters for price, brand, and category constraints

### Natural Language Understanding
- **Brand Recognition:** Automatic identification of brand names in queries
- **Constraint Parsing:** Extract price limits and weight constraints from natural language
  - Price: "under $50", "between $100 and $200"
  - Weight: "lighter than 12kg", "under 15kg"
- **Ambiguity Resolution:** Context-based disambiguation (e.g., "Giant bike" vs "giant screen")

### Semantic Enrichment
- **Rule-based Tagging:** Automatic addition of semantic tags based on product attributes
  - Price-based: budget, affordable, mid-range, premium, luxury
  - Weight-based: light, medium, heavy
  - Category-based: contextual synonyms
- **Benefit:** Users can search "light budget bike" even if these exact words aren't in product descriptions

### Advanced NLP Pipeline
- Tokenization and stop-word removal
- Stemming for root word matching
- POS tagging for better understanding
- Noun phrase extraction
- **NEW:** Dependency parsing for understanding word relationships
- **NEW:** Word embeddings for semantic similarity

### Advanced Features (NEW)
- **Dependency Parsing (spaCy):**
  - **Document-side expansion:** Extracts dependency features during indexing
  - **Query-side expansion:** Enhances queries with grammatical relationships
  - **Vocabulary matching:** Both sides use the same relation tokens
  - Understands grammatical relationships: "bike **with** child seat" → "bike_with_child_seat"
  - Extracts compound terms: "carbon fiber frame" → "carbon fiber", "fiber frame"
  - Identifies prepositional relationships and key concepts
  - **Example:** Query "bike with child seat" matches document containing "bike with child seat" via shared dependency tokens
  
- **Word Embeddings (Sentence Transformers):**
  - Semantic query expansion: "cheap" → "budget", "affordable", "economical"
  - Deep semantic matching: "hills" → "mountain", "climbing", "steep"
  - Hybrid scoring: TF-IDF + semantic similarity
  - Vector search in Elasticsearch

See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) for detailed documentation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                            │
│              "cheap Nike running shoes"                  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  NLP Parser                              │
│  • Extract brands: ["Nike"]                              │
│  • Parse constraints: price_max=50                       │
│  • Clean query: "running shoes"                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Text Preprocessor                           │
│  • Tokenize, remove stop words, stem                     │
│  • Output: "run shoe"                                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Search Indexer                              │
│  • Apply filters (brand, price)                          │
│  • TF-IDF vectorization                                  │
│  • Cosine similarity ranking                             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Ranked Results                          │
│  1. Nike Revolution 6 (score: 0.87)                      │
│  2. Nike Air Zoom (score: 0.72)                          │
└─────────────────────────────────────────────────────────┘
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd fmiunisofia-semantic-search
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data and spaCy model:**
```bash
# NLTK data (downloaded automatically on first run)
# spaCy model (required for dependency parsing)
python -m spacy download en_core_web_sm
```

### Advanced Features Installation

For dependency parsing and word embeddings features:
```bash
# Already included in requirements.txt
pip install spacy sentence-transformers gensim torch

# Download spaCy language model
python -m spacy download en_core_web_sm
```

## Usage

### Flask Web Interface (Recommended)

**Main Entry Point:** `app.py`

The easiest way to use and explore the search engine is through the web interface:

```bash
# Quick start
python app.py

# Or use startup scripts
./start-web.sh    # Linux/Mac
start-web.bat     # Windows
```

Then open your browser to: **http://localhost:5000**

**Features:**
- Configure search backend (Local TF-IDF vs Elasticsearch)
- Toggle advanced NLP features (Dependency Parsing, Word Embeddings)
- Interactive search with real-time results
- Statistics dashboard
- Compare different configurations side-by-side

**Documentation:**
- [WEB_README.md](WEB_README.md) - Complete web interface guide
- [WEB_QUICK_START.md](WEB_QUICK_START.md) - Quick reference
- [WEB_SUMMARY.md](WEB_SUMMARY.md) - Summary and examples

### Command-Line Interface

Run the search engine with generated sample data:

```bash
python src/main.py
```

This will:
1. Generate sample product data (20 products)
2. Build the search index
3. Offer options for demo queries or interactive search

### Using Advanced Features

Enable dependency parsing and word embeddings:

```python
from src.search_engine import ProductSearchEngine

# Initialize with advanced features
engine = ProductSearchEngine(
    use_elasticsearch=True,
    use_dependency_parsing=True,  # Enable spaCy dependency parsing
    use_embeddings=True,           # Enable semantic embeddings
    hybrid_alpha=0.6               # 60% TF-IDF, 40% embeddings
)

# Load and index your data
engine.load_data("data/products.csv")
engine.build_index()

# Search with enhanced understanding
results = engine.search(
    query="cheap bike for hills",
    top_k=10,
    verbose=True  # Shows query enhancement details
)
```

**Example Queries Improved by Advanced Features:**
- "bike with child seat" → Understands relationship structure
- "cheap bike for hills" → Expands to "budget affordable mountain climbing"
- "lightweight racing bike" → Semantic match with "carbon fiber road bike"

Run the advanced examples:
```bash
python examples_advanced.py
```

### Using Your Own Data

Provide a path to your CSV file:

```bash
python src/main.py path/to/your/products.csv
```

**Expected CSV format:**
- Required columns: `title`, `final_price`, `description`, `brand`
- Optional columns: `category`, `weight`

### Interactive Search Examples

```
Search> cheap running shoes
# Returns budget running shoes

Search> premium laptop Dell
# Returns Dell laptops in premium price range

Search> light mountain bike under 1000
# Returns lightweight MTBs under $1000

Search> bike with child seat
# Returns family bikes with child seats
```

## Key Components

### 1. TextPreprocessor (`preprocessing.py`)
Handles all text preprocessing:
- Cleaning and normalization
- Tokenization
- Stop-word removal
- Stemming
- POS tagging

### 2. NLPParser (`nlp_parser.py`)
Parses natural language queries:
- Brand recognition using dictionary lookup
- Price constraint extraction (under, over, between)
- Ambiguous term resolution
- Query cleaning

### 3. SemanticEnricher (`semantic_enrichment.py`)
Adds semantic tags to products:
- Price-based categorization (budget to luxury)
- Weight-based tags (light to heavy)
- Category-specific synonyms

### 4. SearchIndexer (`indexer.py`)
Core search functionality:
- TF-IDF vectorization with bi-grams
- Cosine similarity computation
- Parametric filtering (brand, price, category)
- Result ranking

### 5. ProductSearchEngine (`search_engine.py`)
Orchestrates all components:
- Data loading and normalization
- Index building pipeline
- Query processing and result formatting
- Hybrid scoring with embeddings

### 6. DependencyParser (`dependency_parser.py`)
Advanced linguistic analysis using spaCy:
- Noun phrase extraction
- Compound term detection
- Prepositional relationship extraction
- Query structure analysis

### 7. SemanticEmbeddings (`word_embeddings.py`)
Semantic similarity using transformer models:
- Query expansion with synonyms
- Semantic similarity computation
- Hybrid scoring (TF-IDF + embeddings)
- Domain-specific semantic mappings

## Technical Details

### Search Algorithm

The system uses multiple scoring approaches:

**1. Standard TF-IDF with Cosine Similarity:**
```
similarity(query, product) = (query · product) / (||query|| × ||product||)
```

**2. Hybrid Scoring (with embeddings):**
```
final_score = α × tfidf_score + (1-α) × embedding_similarity
```
Where α (default 0.6) controls the balance between keyword matching and semantic similarity.

**3. Vector Search (Elasticsearch):**
```
script_score = cosineSimilarity(query_vector, content_vector) + text_search_score
```

### TF-IDF Weighting

Term importance is calculated using:
- **Term Frequency (TF):** How often a term appears in a document
- **Inverse Document Frequency (IDF):** How rare a term is across all documents
- **Formula:** `tfidf(t,d) = tf(t,d) × log(N/df(t))`

### Semantic Enrichment Rules

Example rules:
```python
# Price-based
if price < 30: add_tags("budget affordable cheap")
elif price > 300: add_tags("luxury premium high-end")

# Weight-based (for bikes)
if weight < 12kg: add_tags("light lightweight")
```

## Configuration

Edit `src/config.py` to customize:
- Language settings
- Price thresholds for categorization
- Known brand list
- Search parameters (top-k, minimum score)

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: TF-IDF vectorization and metrics
- **nltk**: Natural language processing
- **Flask**: Web framework for UI
- **Flask-CORS**: Cross-origin support for web API
- **spacy**: Advanced NLP and dependency parsing
- **sentence-transformers**: Semantic embeddings
- **elasticsearch**: Distributed search backend
- **torch**: Deep learning framework for embeddings

## Example Queries and Results

| Query | Extracted Info | Top Results |
|-------|---------------|-------------|
| "cheap running shoes" | price: cheap | Budget athletic shoes |
| "Trek mountain bike under 1000" | brand: Trek, price_max: 1000 | Trek Marlin 7 |
| "light bike for trails" | weight: light, context: trails | Lightweight MTBs |
| "bike lighter than 12kg" | weight_max: 12kg | Bikes under 12kg |
| "lightweight carbon bike under 8kg" | weight_max: 8kg, keywords: lightweight carbon | Ultra-light carbon bikes |
| "premium laptop for work" | price: premium, context: work | High-end business laptops |

## Comparing Search Backends

The web interface lets you easily compare different search configurations:

| Configuration | Speed | Accuracy | Best For |
|--------------|-------|----------|----------|
| **Local TF-IDF** | Very Fast | Good | Testing, keyword search |
| **Elasticsearch** | Fast | Very Good | Production, large datasets |
| **+ Word Embeddings** | Moderate | Excellent | Semantic understanding |
| **+ Dependency Parsing** | Moderate | Outstanding | Complex NLP queries |

**Example:** Query = "affordable footwear for jogging"
- **TF-IDF**: Looks for exact words "affordable", "footwear", "jogging"
- **+ Embeddings**: Also finds "cheap shoes" and "running" products
- **+ Dependency**: Understands "affordable" modifies "footwear", not "jogging"

Try these configurations yourself through the web interface!

## Future Enhancements

Potential improvements mentioned in DESCRIPTION.md:
- [ ] Elasticsearch integration for production scale
- [ ] Word embeddings (Word2Vec/FastText) for deeper semantic understanding
- [ ] Dependency parsing for complex query structures
- [ ] User feedback learning
- [ ] Category prediction
- [ ] Multi-language support

## License

See LICENSE file for details.

## Author

Built as part of Information Retrieval course project at Sofia University Faculty of Mathematics and Informatics.

## References

Based on concepts from:
- Vector Space Model (Salton & McGill, 1983)
- TF-IDF (Sparck Jones, 1972)
- Cosine Similarity for information retrieval
