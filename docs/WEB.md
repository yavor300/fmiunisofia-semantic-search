# Flask Web Interface - Complete Guide

## Overview

This is a **Flask web application** that provides a interface for the Semantic Search Engine.
You can easily configure and compare different search backends (Local TF-IDF vs Elasticsearch) and toggle advanced NLP features through your browser.

---

## Main Entry Point

**`app.py`** is the main entry point for the web application.

To start the application:
```bash
python app.py
```

Or use the convenience scripts:
- Linux/Mac: `./start-web.sh`
- Windows: `start-web.bat`

Then open your browser to: **http://localhost:5000**

---

## Key Features

### 1. **Dynamic Configuration**
Switch between different configurations without editing code:
- **Local TF-IDF Indexer**
- **Elasticsearch**
- **Word Embeddings**
- **Dependency Parsing**

### 2. **Real-time Comparison**
Test the same query with different configurations to see how each component affects results:

**Example: Query = "cheap running shoes"**

| Configuration | What it does |
|--------------|--------------|
| Local TF-IDF | Fast keyword matching on "cheap", "running", "shoes" |
| + Elasticsearch | Better full-text search with BM25 scoring |
| + Word Embeddings | Understands "cheap" ≈ "affordable" ≈ "budget" |
| + Dependency Parsing | Understands "cheap" modifies "shoes", not "running" |

### 3. **Interactive Search**
- Enter natural language queries
- See results with relevance scores
- View product details (price, brand, rating, category)
- Get instant feedback on search quality

### 4. **Statistics Dashboard**
Monitor dataset and search engine statistics:
- Total products indexed
- Number of unique brands
- Price ranges
- Average ratings
- Elasticsearch connection status

---

## Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask & Flask-CORS (web framework)
- pandas, numpy, scikit-learn (data processing)
- nltk, spacy (NLP)
- sentence-transformers (embeddings)
- elasticsearch (optional, for distributed search)

### Step 2: Install spaCy Model (Optional)

For dependency parsing:
```bash
python -m spacy download en_core_web_sm
```

### Step 3: Start Elasticsearch (Optional)

If you want to use Elasticsearch:
```bash
docker compose up -d
```

Wait ~30 seconds for it to start.

### Step 4: Run the Web Application

```bash
python app.py
```

Or use the startup scripts:
```bash
# Linux/Mac
./start-web.sh

# Windows
start-web.bat
```

### Step 5: Open in Browser

Navigate to: **http://localhost:5000**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Browser                          │
│              (http://localhost:5000)                    │
└─────────────────────────────────────────────────────────┘
                          │
                          │ HTTP Requests
                          ↓
┌─────────────────────────────────────────────────────────┐
│                     app.py (Flask)                      │
│                                                         │
│  Routes:                                                │
│  • GET  /               → Render web interface          │
│  • GET  /api/config     → Get configuration             │
│  • POST /api/config     → Update configuration          │
│  • POST /api/initialize → Initialize search engine      │
│  • POST /api/search     → Perform search                │
│  • GET  /api/status     → System status                 │
│  • POST /api/reset      → Reset engine                  │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Calls
                          ↓
┌─────────────────────────────────────────────────────────┐
│            ProductSearchEngine (src/search_engine.py)   │
│                                                         │
│  Components:                                            │
│  • TextPreprocessor    → Clean and tokenize text        │
│  • NLPParser           → Extract brands, prices         │
│  • SemanticEnricher    → Add semantic tags              │
│  • Indexer             → Local or Elasticsearch         │
└─────────────────────────────────────────────────────────┘
                          │
                    ┌─────┴─────┐
                    │           │
                    ↓           ↓
         ┌──────────────┐  ┌─────────────┐
         │   Local      │  │Elasticsearch│
         │  TF-IDF      │  │   Docker    │
         │  Indexer     │  │  Container  │
         └──────────────┘  └─────────────┘
```