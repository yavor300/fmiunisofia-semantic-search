"""
Domain-Specific Intelligent Product Search Engine

An advanced Information Retrieval system combining NLP, semantic enrichment,
and vector space models for e-commerce product search.
"""

from src.search_engine import ProductSearchEngine
from src.preprocessing import TextPreprocessor
from src.nlp_parser import NLPParser
from src.semantic_enrichment import SemanticEnricher
from src.indexer import SearchIndexer

__version__ = "1.0.0"

__all__ = [
    "ProductSearchEngine",
    "TextPreprocessor",
    "NLPParser",
    "SemanticEnricher",
    "SearchIndexer"
]
