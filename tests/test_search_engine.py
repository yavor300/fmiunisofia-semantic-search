"""
Unit tests for the search engine components.
Run with: python -m pytest tests/test_search_engine.py
"""

import pytest
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.nlp_parser import NLPParser
from src.semantic_enrichment import SemanticEnricher
from src.indexer import SearchIndexer


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        text = "Hello World! This is a TEST. #hashtag @mention"
        result = self.preprocessor.clean_text(text)
        assert result == "hello world this is a test hashtag mention"
    
    def test_preprocess_full(self):
        text = "Running shoes for the best runners"
        result = self.preprocessor.preprocess(text)
        # Should be stemmed and stop words removed
        assert "run" in result
        assert "shoe" in result
        assert "the" not in result
    
    def test_empty_text(self):
        result = self.preprocessor.preprocess("")
        assert result == ""


class TestNLPParser:
    """Test NLP parsing functionality."""
    
    def setup_method(self):
        self.parser = NLPParser()
    
    def test_extract_brands(self):
        query = "I want Nike running shoes"
        brands = self.parser.extract_brands(query)
        assert "nike" in brands
    
    def test_extract_price_under(self):
        query = "laptop under 500"
        constraints = self.parser.extract_price_constraints(query)
        assert "max" in constraints
        assert constraints["max"] == 500
    
    def test_extract_price_over(self):
        query = "bike over 1000 dollars"
        constraints = self.parser.extract_price_constraints(query)
        assert "min" in constraints
        assert constraints["min"] == 1000
    
    def test_extract_price_between(self):
        query = "laptop between 500 and 1000"
        constraints = self.parser.extract_price_constraints(query)
        assert constraints["min"] == 500
        assert constraints["max"] == 1000
    
    def test_parse_query(self):
        query = "cheap Nike shoes under 100"
        result = self.parser.parse_query(query)
        
        assert "nike" in result["brands"]
        assert result["price_constraints"]["max"] == 100
        assert "cheap" in result["clean_query"]


class TestSemanticEnricher:
    """Test semantic enrichment functionality."""
    
    def setup_method(self):
        self.enricher = SemanticEnricher()
    
    def test_price_category_budget(self):
        category = self.enricher.get_price_category(25)
        assert category == "budget"
    
    def test_price_category_premium(self):
        category = self.enricher.get_price_category(200)
        assert category == "premium"
    
    def test_price_category_luxury(self):
        category = self.enricher.get_price_category(500)
        assert category == "luxury"
    
    def test_enrich_product(self):
        product = {
            "price_numeric": 25,
            "weight": 10,
            "category": "bike"
        }
        tags = self.enricher.enrich_product(product)
        assert "budget" in tags
        assert "light" in tags


class TestSearchIndexer:
    """Test search indexing functionality."""
    
    def setup_method(self):
        # Create sample data
        self.df = pd.DataFrame([
            {
                "name": "Running Shoes",
                "brand": "Nike",
                "price_numeric": 100,
                "searchable_content": "run shoe nike sport athlet"
            },
            {
                "name": "Mountain Bike",
                "brand": "Trek",
                "price_numeric": 800,
                "searchable_content": "mountain bike trek trail offroad"
            },
            {
                "name": "Laptop Computer",
                "brand": "Dell",
                "price_numeric": 1200,
                "searchable_content": "laptop comput dell work productiv"
            }
        ])
        
        self.indexer = SearchIndexer(max_features=100)
        self.indexer.build_index(self.df)
    
    def test_build_index(self):
        assert self.indexer.tfidf_matrix is not None
        assert self.indexer.tfidf_matrix.shape[0] == 3
    
    def test_search_basic(self):
        results = self.indexer.search("run shoe", top_k=2)
        assert len(results) > 0
        assert results.iloc[0]["name"] == "Running Shoes"
    
    def test_search_with_brand_filter(self):
        filters = {"brand": ["Trek"]}
        results = self.indexer.search("bike", filters=filters)
        assert len(results) > 0
        assert results.iloc[0]["brand"] == "Trek"
    
    def test_search_with_price_filter(self):
        filters = {"price_max": 500}
        results = self.indexer.search("shoe", filters=filters)
        assert len(results) > 0
        assert all(row["price_numeric"] <= 500 for _, row in results.iterrows())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
