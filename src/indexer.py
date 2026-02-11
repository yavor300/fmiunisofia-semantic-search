"""
Indexing module using TF-IDF Vector Space Model.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    MAX_FEATURES,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE,
    DEFAULT_MAX_FEATURES
)


class SearchIndexer:
    """Handles indexing and searching using TF-IDF and cosine similarity."""
    
    def __init__(
        self,
        max_features: int = None,
        ngram_range: tuple = None,
        min_df: int = None,
        sublinear_tf: bool = None
    ):
        """
        Initialize the indexer.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorizer (defaults to config.DEFAULT_MAX_FEATURES)
            ngram_range: N-gram range for TF-IDF (defaults to (1, 2) for unigrams and bigrams)
            min_df: Minimum document frequency (defaults to 1)
            sublinear_tf: Use logarithmic TF scaling (defaults to True)
        """
        max_features = max_features if max_features is not None else DEFAULT_MAX_FEATURES
        ngram_range = ngram_range if ngram_range is not None else (1, 2)
        min_df = min_df if min_df is not None else 1
        sublinear_tf = sublinear_tf if sublinear_tf is not None else True
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf
        )
        self.tfidf_matrix = None
        self.df = None
        self.feature_names = None
    
    def build_index(self, df: pd.DataFrame, content_field: str = "searchable_content"):
        """
        Build TF-IDF index from dataframe.
        
        Args:
            df: DataFrame with product data
            content_field: Name of the field containing preprocessed content
        """
        if df.empty:
            raise ValueError("Cannot build index on empty dataframe")
        
        if content_field not in df.columns:
            raise ValueError(f"Content field '{content_field}' not found in dataframe")
        
        df = df[df[content_field].str.len() > 0].copy()
        
        if df.empty:
            raise ValueError("No valid content to index after filtering")
        
        self.df = df
        
        print(f"Building TF-IDF index for {len(df)} products...")
        self.tfidf_matrix = self.vectorizer.fit_transform(df[content_field])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Index built successfully. Shape: {self.tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.feature_names)}")
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY_SCORE,
        filters: dict = None
    ) -> pd.DataFrame:
        """
        Search for products matching the query.
        
        Args:
            query: Preprocessed search query
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            filters: Dictionary of filters to apply (e.g., {"brand": "nike", "price_max": 100})
            
        Returns:
            DataFrame with search results including similarity scores
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if not query:
            return pd.DataFrame()
        
        filtered_df = self.df
        filtered_indices = np.arange(len(self.df))
        
        if filters:
            filtered_df, filtered_indices = self._apply_filters(filters)
            if filtered_df.empty:
                return pd.DataFrame()
        
        query_vector = self.vectorizer.transform([query])
        
        filtered_matrix = self.tfidf_matrix[filtered_indices]
        similarities = cosine_similarity(query_vector, filtered_matrix).flatten()
        
        score_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in score_indices[:top_k]:
            score = similarities[idx]
            if score >= min_score:
                original_idx = filtered_indices[idx]
                product = filtered_df.iloc[idx].copy()
                product["similarity_score"] = round(score, 4)
                results.append(product)
        
        if not results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def _apply_filters(self, filters: dict):
        """
        Apply parametric filters to narrow down search space.
        
        Args:
            filters: Dictionary with filter conditions
            
        Returns:
            Tuple of (filtered_df, filtered_indices)
        """
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        if "brand" in filters:
            brands = filters["brand"]
            if isinstance(brands, str):
                brands = [brands]
            mask &= self.df["brand"].str.lower().isin([b.lower() for b in brands])
        
        if "price_min" in filters:
            mask &= self.df["price_numeric"] >= filters["price_min"]
        
        if "price_max" in filters:
            mask &= self.df["price_numeric"] <= filters["price_max"]
        
        if "weight" in self.df.columns:
            if "weight_min" in filters:
                mask &= self.df["weight"] >= filters["weight_min"]
            
            if "weight_max" in filters:
                mask &= self.df["weight"] <= filters["weight_max"]
        
        if "category" in filters and "category" in self.df.columns:
            categories = filters["category"]
            if isinstance(categories, str):
                categories = [categories]
            mask &= self.df["category"].str.lower().isin([c.lower() for c in categories])
        
        filtered_df = self.df[mask].copy()
        filtered_indices = np.where(mask)[0]
        
        return filtered_df, filtered_indices
