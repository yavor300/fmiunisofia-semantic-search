# Elasticsearch Integration Guide

This guide explains how to use the semantic search engine with Elasticsearch and product data.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                              │
│              "cheap running shoes"                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              NLP Query Parser                               │
│  • Extract brands, constraints                              │
│  • Clean and normalize query                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Text Preprocessor + Semantic Tags                   │
│  • Tokenize, stem, remove stopwords                         │
│  • Add semantic enrichment                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              ELASTICSEARCH                                  │
│  ┌───────────────────────────────────────────┐              │
│  │  Index: amazon_products                   │              │
│  │  • Full-text search                       │              │
│  │  • Custom analyzers (stemming, stopwords) │              │
│  │  • Multi-field matching                   │              │
│  │  • Parametric filtering (price, brand)    │              │
│  │  • Relevance scoring (BM25)               │              │
│  └───────────────────────────────────────────┘              │
│           Running in Docker Container                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Ranked Results                             │
│  1. Saucony Running Shoes (score: 8.45)                     │
│  2. Nike Revolution (score: 7.23)                           │
└─────────────────────────────────────────────────────────────┘
```

### Document Model

Each product is treated as a **document** in the corpus:

```json
{
  "product_id": "12345",
  "title": "Saucony Men's Kinvara 13 Running Shoe",
  "brand": "Saucony",
  "description": "Lightweight running shoe...",
  "price": 57.79,
  "category": "Running Shoes",
  "semantic_tags": "affordable mid-range lightweight",
  "searchable_content": "saucony running shoe lightweight...",
  "rating": 4.6,
  "reviews_count": 702,
  "availability": "In Stock"
}
```
