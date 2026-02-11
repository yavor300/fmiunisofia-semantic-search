# Installation Guide: Advanced Features

This guide will help you set up the advanced features (dependency parsing and word embeddings) for the semantic search engine.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- At least 2GB of free RAM (for loading models)
- Internet connection (for downloading models)

## Step-by-Step Installation

### 1. Install Python Dependencies

First, ensure you're in the project directory, then install all required packages:

```bash
cd fmiunisofia-semantic-search
pip install -r requirements.txt
```

This will install:
- `spacy>=3.7.0` - For dependency parsing
- `sentence-transformers>=2.2.0` - For semantic embeddings
- `gensim>=4.3.0` - Word embedding support
- `torch>=2.0.0` - Deep learning backend
- All existing dependencies (pandas, nltk, elasticsearch, etc.)

### 2. Download spaCy Language Model

The dependency parsing feature requires a spaCy language model:

```bash
python -m spacy download en_core_web_sm
```
