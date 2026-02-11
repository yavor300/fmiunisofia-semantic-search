# Advanced Features: Dependency Parsing & Word Embeddings

This document describes the newly integrated advanced NLP features for semantic search.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Dependency Parsing](#dependency-parsing)
4. [Word Embeddings](#word-embeddings)
5. [Usage Examples](#usage-examples)
6. [Performance Considerations](#performance-considerations)

---

## Overview

The semantic search engine has been enhanced with two powerful NLP features:

### 1. **Dependency Parsing** (using spaCy)
- Understands grammatical relationships between words
- Extracts compound terms and prepositional relationships
- Improves query understanding for complex phrases

**Example:**
- Query: "bike with child seat"
- Understanding: Recognizes "bike" as head noun, "with" as relationship, "child seat" as object
- Result: Better matching for bikes that actually include child seats

### 2. **Word Embeddings** (using Sentence Transformers)
- Captures semantic similarity beyond exact word matches
- Enables query expansion with synonyms
- Provides hybrid scoring combining TF-IDF with semantic similarity

**Example:**
- Query: "cheap bike for hills"
- Expansions: "budget", "affordable", "economical" (for "cheap")
             "mountain", "climbing", "steep" (for "hills")
- Result: Finds relevant products even without exact keyword matches

---
