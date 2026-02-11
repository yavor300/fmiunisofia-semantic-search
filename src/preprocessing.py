"""
Text preprocessing module for NLP pipeline.
Handles tokenization, stop-word removal, stemming, and advanced dependency parsing.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag, word_tokenize
from typing import Dict, List, Optional
import logging

from src.config import (
    LANGUAGE,
    DEFAULT_LANGUAGE,
    DEFAULT_USE_DEPENDENCY_PARSING
)

logger = logging.getLogger(__name__)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)


class TextPreprocessor:
    """
    Handles all text preprocessing operations.
    
    Features:
        - Basic preprocessing (cleaning, tokenization, stemming)
        - Advanced dependency parsing integration
        - Noun phrase extraction
        - Query enhancement for better search results
    """
    
    def __init__(
        self,
        language: str = None,
        use_dependency_parsing: bool = None
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            language: Language for stopwords and stemming (defaults to config.DEFAULT_LANGUAGE)
            use_dependency_parsing: Enable advanced dependency parsing (defaults to config.DEFAULT_USE_DEPENDENCY_PARSING)
        """
        language = language if language is not None else DEFAULT_LANGUAGE
        use_dependency_parsing = use_dependency_parsing if use_dependency_parsing is not None else DEFAULT_USE_DEPENDENCY_PARSING
        
        self.stemmer = SnowballStemmer(language)
        self.stop_words = set(stopwords.words(language))
        self.use_dependency_parsing = use_dependency_parsing
        
        self.dependency_parser = None
        if use_dependency_parsing:
            try:
                from src.dependency_parser import DependencyParser
                self.dependency_parser = DependencyParser()
                logger.info("Dependency parsing enabled")
            except Exception as e:
                logger.warning(f"Could not initialize dependency parser: {e}")
                logger.warning("Falling back to basic preprocessing")
                self.use_dependency_parsing = False
    
    def clean_text(self, text):
        """Basic text cleaning: lowercase and remove special characters."""
        if not text or not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s-]", "", text)
        return text
    
    def tokenize(self, text):
        """Tokenize text into words."""
        if not text:
            return []
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove common stop words from token list."""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to reduce words to root form."""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text, remove_stops=True, stem=True):
        """
        Full preprocessing pipeline: clean, tokenize, remove stops, stem.
        
        Args:
            text: Input text string
            remove_stops: Whether to remove stop words
            stem: Whether to apply stemming
            
        Returns:
            Preprocessed text as a single string
        """
        if not text:
            return ""
        
        text = self.clean_text(text)
        
        tokens = text.split()
        
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        if stem:
            tokens = self.stem_tokens(tokens)
        
        return " ".join(tokens)
    
    def preprocess_enhanced_query(self, enhanced_query: str, original_query: str) -> str:
        """
        Preprocess an enhanced query while preserving enhancement tags.
        
        Tags with underscores (e.g., 'mountain_bike', 'bike_with_child_seat') are
        preserved as-is, while the original query terms are stemmed normally.
        
        Args:
            enhanced_query: Query with added enhancement tags
            original_query: Original query before enhancement
            
        Returns:
            Preprocessed query with preserved tags
        """
        if not enhanced_query:
            return ""
        
        tokens = enhanced_query.split()
        
        processed_tokens = []
        for token in tokens:

            if '_' in token:

                processed_tokens.append(token)
            else:

                cleaned = self.clean_text(token)
                if cleaned and cleaned not in self.stop_words:
                    stemmed = self.stemmer.stem(cleaned)
                    if stemmed:
                        processed_tokens.append(stemmed)
        
        return " ".join(processed_tokens)

    def enhance_query(self, query: str) -> str:
        """
        Enhance query by adding extracted relationships and compound terms.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query with additional terms
        """
        if self.use_dependency_parsing and self.dependency_parser:
            return self.dependency_parser.enhance_query_for_search(query)
        
        return self.preprocess(query)
    
    def extract_dependency_features(self, text: str) -> str:
        """
        Extract dependency parsing features for document indexing.
        Adds compound terms, relationships, and key concepts as additional tokens.
        
        This ensures documents and queries share the same vocabulary space.
        
        Args:
            text: Input text (document content)
            
        Returns:
            Space-separated string of additional dependency tokens
        """
        if not self.use_dependency_parsing or not self.dependency_parser:
            return ""
        
        try:
            analysis = self.dependency_parser.analyze_query_structure(text)
            
            features = []
            
            for term in analysis.get("compound_terms", []):
                tag_term = term.replace(" ", "_")
                features.append(tag_term)
            for rel in analysis.get("relationships", []):
                if "relation" in rel:
                    features.append(rel["relation"])

            for concept in analysis.get("key_concepts", []):
                tag_concept = concept.replace(" ", "_")
                features.append(tag_concept)

            unique_features = list(dict.fromkeys(features))
            
            return " ".join(unique_features)
            
        except Exception as e:
            logger.warning(f"Error extracting dependency features: {e}")
            return ""
