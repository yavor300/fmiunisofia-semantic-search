"""
Semantic enrichment module.
Adds hidden semantic tags to documents and queries based on their attributes.
Enhanced with word embeddings for deep semantic understanding.
Supports dynamic tag generation using WordNet and word embeddings.
"""

import pandas as pd
from typing import Dict, List, Optional, Set
import logging

from src.config import (
    PRICE_THRESHOLDS,
    PRICE_TAGS,
    WEIGHT_THRESHOLDS,
    WEIGHT_TAGS,
    DYNAMIC_TAGS_ENABLED,
    USE_EMBEDDINGS_FOR_TAGS,
    USE_WORDNET_FOR_TAGS,
    CACHE_DYNAMIC_TAGS,
    MAX_SYNONYMS_PER_SEED,
    PRICE_TAG_SEEDS,
    WEIGHT_TAG_SEEDS,
    CATEGORY_TAG_SEEDS
)

logger = logging.getLogger(__name__)


class DynamicTagGenerator:
    """
    Dynamically generates semantic tags using word embeddings and WordNet.
    
    Features:
        - Uses word embeddings for high-quality semantic similarity
        - Falls back to WordNet for synonym generation
        - Caches generated tags for performance
        - Configurable via config.py
    """
    
    def __init__(
        self,
        use_embeddings: bool = None,
        use_wordnet: bool = None,
        cache_tags: bool = None,
        max_synonyms: int = None
    ):
        """
        Initialize the dynamic tag generator.
        
        Args:
            use_embeddings: Whether to use word embeddings (defaults to config.USE_EMBEDDINGS_FOR_TAGS)
            use_wordnet: Whether to use WordNet (defaults to config.USE_WORDNET_FOR_TAGS)
            cache_tags: Whether to cache generated tags (defaults to config.CACHE_DYNAMIC_TAGS)
            max_synonyms: Maximum synonyms per seed word (defaults to config.MAX_SYNONYMS_PER_SEED)
        """

        from src.config import (
            DEFAULT_USE_EMBEDDINGS_FOR_TAGS,
            DEFAULT_USE_WORDNET_FOR_TAGS,
            DEFAULT_CACHE_DYNAMIC_TAGS,
            DEFAULT_MAX_SYNONYMS_PER_SEED
        )
        
        self.use_embeddings = use_embeddings if use_embeddings is not None else DEFAULT_USE_EMBEDDINGS_FOR_TAGS
        self.use_wordnet = use_wordnet if use_wordnet is not None else DEFAULT_USE_WORDNET_FOR_TAGS
        self.cache_tags = cache_tags if cache_tags is not None else DEFAULT_CACHE_DYNAMIC_TAGS
        self.max_synonyms = max_synonyms if max_synonyms is not None else DEFAULT_MAX_SYNONYMS_PER_SEED
        self._cached_tags: Dict[str, str] = {}
        
        self.embeddings = None
        if self.use_embeddings:
            try:
                from src.word_embeddings import SemanticEmbeddings
                self.embeddings = SemanticEmbeddings()
                logger.info("Word embeddings enabled for dynamic tag generation")
            except Exception as e:
                logger.warning(f"Could not initialize embeddings for tags: {e}")
                self.embeddings = None
        
        self.wordnet = None
        if self.use_wordnet:
            try:
                import nltk
                from nltk.corpus import wordnet
                try:
                    wordnet.synsets('test')
                except LookupError:
                    logger.info("Downloading WordNet data...")
                    nltk.download('wordnet', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                self.wordnet = wordnet
                logger.info("WordNet enabled for dynamic tag generation")
            except Exception as e:
                logger.warning(f"Could not initialize WordNet for tags: {e}")
                self.wordnet = None
    
    def get_synonyms_wordnet(self, word: str, max_synonyms: int) -> List[str]:
        """
        Get synonyms using WordNet.
        
        Args:
            word: Word to find synonyms for
            max_synonyms: Maximum number of synonyms
            
        Returns:
            List of synonyms
        """
        if not self.wordnet:
            return []
        
        synonyms: Set[str] = set()
        try:
            for syn in self.wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', '-').lower()
                    if synonym != word.lower() and len(synonym.split('-')) <= 3:
                        synonyms.add(synonym)
                    if len(synonyms) >= max_synonyms:
                        break
                if len(synonyms) >= max_synonyms:
                    break
        except Exception as e:
            logger.debug(f"Error getting WordNet synonyms for '{word}': {e}")
        
        return list(synonyms)
    
    def get_synonyms_embeddings(self, word: str, max_synonyms: int) -> List[str]:
        """
        Get similar words using word embeddings.
        
        Args:
            word: Word to find similar words for
            max_synonyms: Maximum number of similar words
            
        Returns:
            List of similar words
        """
        if not self.embeddings:
            return []
        
        try:
            # Use the expand_query method which already has domain knowledge
            similar_words = self.embeddings.expand_query(
                word,
                max_expansions=max_synonyms
            )
            return similar_words
        except Exception as e:
            logger.debug(f"Error getting embedding synonyms for '{word}': {e}")
            return []
    
    def generate_tags(
        self,
        category: str,
        seed_words: List[str],
        max_per_seed: Optional[int] = None
    ) -> str:
        """
        Generate dynamic tags for a category.
        
        Args:
            category: Category name (e.g., 'budget', 'light')
            seed_words: Base words to expand from
            max_per_seed: Maximum expansions per seed word
            
        Returns:
            Space-separated string of tags
        """
        if max_per_seed is None:
            max_per_seed = self.max_synonyms
        
        cache_key = f"{category}_{','.join(sorted(seed_words))}_{max_per_seed}"
        if self.cache_tags and cache_key in self._cached_tags:
            return self._cached_tags[cache_key]
        
        all_tags: Set[str] = set(seed_words)
        
        for seed in seed_words:
            if self.embeddings:
                synonyms = self.get_synonyms_embeddings(seed, max_per_seed)
                all_tags.update(synonyms)
            
            if self.wordnet:
                synonyms = self.get_synonyms_wordnet(seed, max_per_seed)
                all_tags.update(synonyms)
        
        result = " ".join(sorted(all_tags))
        
        if self.cache_tags:
            self._cached_tags[cache_key] = result
        
        return result
    
    def generate_all_price_tags(self) -> Dict[str, str]:
        """Generate all price category tags dynamically."""
        return {
            category: self.generate_tags(category, seeds)
            for category, seeds in PRICE_TAG_SEEDS.items()
        }
    
    def generate_all_weight_tags(self) -> Dict[str, str]:
        """Generate all weight category tags dynamically."""
        return {
            category: self.generate_tags(category, seeds)
            for category, seeds in WEIGHT_TAG_SEEDS.items()
        }
    
    def generate_category_tags(self, category_key: str) -> str:
        """
        Generate tags for a specific category.
        
        Args:
            category_key: Category keyword (e.g., 'bike', 'laptop')
            
        Returns:
            Space-separated tags
        """
        if category_key.lower() in CATEGORY_TAG_SEEDS:
            seeds = CATEGORY_TAG_SEEDS[category_key.lower()]
            return self.generate_tags(category_key, seeds)
        return ""
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached tags."""
        return {
            "cache_enabled": self.cache_tags,
            "cached_entries": len(self._cached_tags) if self.cache_tags else 0,
            "embeddings_enabled": self.embeddings is not None,
            "wordnet_enabled": self.wordnet is not None
        }
    
    def clear_cache(self):
        """Clear the tag cache."""
        if self.cache_tags:
            self._cached_tags.clear()
            logger.info("Dynamic tag cache cleared")


class SemanticEnricher:
    """
    Enriches product data with semantic tags based on rules and embeddings.
    
    Features:
        - Rule-based semantic tagging (price, weight, category)
        - Word embedding-based semantic expansion
        - Query enhancement using semantic similarity
        - Dependency parsing-based feature extraction (NEW)
    """
    
    def __init__(
        self,
        use_embeddings: bool = None,
        use_dynamic_tags: bool = None,
        price_thresholds: Dict = None,
        weight_thresholds: Dict = None,
        preprocessor = None
    ):
        """
        Initialize the semantic enricher.
        
        Args:
            use_embeddings: Enable word embedding-based enrichment (defaults to config.DEFAULT_USE_EMBEDDINGS_FOR_ENRICHMENT)
            use_dynamic_tags: Enable dynamic tag generation (defaults to config.DYNAMIC_TAGS_ENABLED)
            price_thresholds: Custom price thresholds (defaults to config.PRICE_THRESHOLDS)
            weight_thresholds: Custom weight thresholds (defaults to config.WEIGHT_THRESHOLDS)
            preprocessor: TextPreprocessor instance for dependency parsing (optional)
        """
        from src.config import (
            DEFAULT_USE_EMBEDDINGS_FOR_ENRICHMENT,
            DYNAMIC_TAGS_ENABLED
        )
        
        self.price_thresholds = price_thresholds if price_thresholds is not None else PRICE_THRESHOLDS
        self.weight_thresholds = weight_thresholds if weight_thresholds is not None else WEIGHT_THRESHOLDS
        self.use_embeddings = use_embeddings if use_embeddings is not None else DEFAULT_USE_EMBEDDINGS_FOR_ENRICHMENT
        self.use_dynamic_tags = use_dynamic_tags if use_dynamic_tags is not None else DYNAMIC_TAGS_ENABLED
        self.preprocessor = preprocessor
        
        self.tag_generator = None
        if self.use_dynamic_tags:
            try:
                self.tag_generator = DynamicTagGenerator()
                self.price_tags = self.tag_generator.generate_all_price_tags()
                self.weight_tags = self.tag_generator.generate_all_weight_tags()
                logger.info("Dynamic tag generation enabled")
                logger.info(f"Tag generator stats: {self.tag_generator.get_cache_stats()}")
            except Exception as e:
                logger.warning(f"Could not initialize dynamic tag generator: {e}")
                logger.warning("Falling back to static tags")
                self.price_tags = PRICE_TAGS
                self.weight_tags = WEIGHT_TAGS
                self.use_dynamic_tags = False
        else:
            self.price_tags = PRICE_TAGS
            self.weight_tags = WEIGHT_TAGS
        
        self.embeddings = None
        if self.use_embeddings:
            try:
                from src.word_embeddings import SemanticEmbeddings
                self.embeddings = SemanticEmbeddings()
                logger.info("Word embeddings enabled for semantic enrichment")
            except Exception as e:
                logger.warning(f"Could not initialize word embeddings: {e}")
                logger.warning("Falling back to rule-based enrichment only")
                self.use_embeddings = False
    
    def get_price_category(self, price: float) -> str:
        """
        Determine price category for a given price.
        
        Args:
            price: Numeric price value
            
        Returns:
            Category name (budget, affordable, mid_range, premium, luxury)
        """
        for category, (min_price, max_price) in self.price_thresholds.items():
            if min_price <= price < max_price:
                return category
        return "unknown"
    
    def get_weight_category(self, weight: float) -> str:
        """
        Determine weight category for a given weight.
        
        Args:
            weight: Numeric weight value in kg
            
        Returns:
            Category name (light, medium, heavy)
        """
        for category, (min_weight, max_weight) in self.weight_thresholds.items():
            if min_weight <= weight < max_weight:
                return category
        return "unknown"
    
    def enrich_product(self, product: Dict) -> str:
        """
        Enrich a single product with semantic tags.
        
        Args:
            product: Dictionary with product attributes (price_numeric, weight, etc.)
            
        Returns:
            Space-separated string of semantic tags
        """
        tags = []
        
        if "price_numeric" in product and product["price_numeric"] > 0:
            price_cat = self.get_price_category(product["price_numeric"])
            if price_cat != "unknown" and price_cat in self.price_tags:
                tags.append(self.price_tags[price_cat])
        
        if "weight" in product and product["weight"] > 0:
            weight_cat = self.get_weight_category(product["weight"])
            if weight_cat != "unknown" and weight_cat in self.weight_tags:
                tags.append(self.weight_tags[weight_cat])
        
        if "category" in product:
            category_tags = self._get_category_tags(product["category"])
            if category_tags:
                tags.append(category_tags)
        
        return " ".join(tags)
    
    def _get_category_tags(self, category: str) -> str:
        """
        Get additional semantic tags based on product category.
        Enhanced with dynamic tag generation and embedding-based expansion.
        
        Args:
            category: Product category
            
        Returns:
            Space-separated semantic tags
        """
        category_lower = category.lower() if category else ""
        
        tags = []
        
        if self.use_dynamic_tags and self.tag_generator:
            for key in CATEGORY_TAG_SEEDS.keys():
                if key in category_lower:
                    dynamic_tags = self.tag_generator.generate_category_tags(key)
                    if dynamic_tags:
                        tags.append(dynamic_tags)
        else:
            category_mappings = {
                "bike": "bicycle cycling rider",
                "mtb": "mountain off-road trail",
                "road": "racing speed performance",
                "hybrid": "versatile commuting city",
                "shoes": "footwear sneakers",
                "running": "jogging athletic fitness",
                "laptop": "computer notebook portable",
                "phone": "smartphone mobile device",
                "headphones": "audio earphones listening music",
                "camera": "photography video shooting",
            }
            
            for key, mapping_tags in category_mappings.items():
                if key in category_lower:
                    tags.append(mapping_tags)
        
        if self.use_embeddings and self.embeddings and category_lower:
            try:
                expanded = self.embeddings.expand_query(
                    category_lower,
                    max_expansions=2
                )
                if expanded:
                    tags.append(" ".join(expanded))
            except Exception as e:
                logger.debug(f"Could not expand category: {e}")
        
        return " ".join(tags)
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich all products in a dataframe.
        
        Args:
            df: DataFrame with product data
            
        Returns:
            DataFrame with added 'semantic_tags' and 'dependency_features' columns
        """
        if df.empty:
            return df
        
        semantic_tags = []
        dependency_features = []
        
        for _, row in df.iterrows():
            tags = self.enrich_product(row.to_dict())
            semantic_tags.append(tags)
            
            dep_features = self._extract_document_dependency_features(row)
            dependency_features.append(dep_features)
        
        df["semantic_tags"] = semantic_tags
        df["dependency_features"] = dependency_features
        return df
    
    def _extract_document_dependency_features(self, row) -> str:
        """
        Extract dependency parsing features from a product document.
        
        This extracts compound terms, relationships, and key concepts from
        the product's name and description, creating the same vocabulary
        space as the query-side dependency parsing.
        
        Args:
            row: Product row with name, desc fields
            
        Returns:
            Space-separated string of dependency features
        """
        if not self.preprocessor:
            return ""
        
        all_features = []
        
        if "name" in row and row["name"]:
            title = str(row["name"])
            title_features = self.preprocessor.extract_dependency_features(title)
            if title_features:
                all_features.append(title_features)
        
        if "desc" in row and row["desc"]:
            desc = str(row["desc"])[:500]
            desc_features = self.preprocessor.extract_dependency_features(desc)
            if desc_features:
                all_features.append(desc_features)
        
        return " ".join(all_features)
    
    def create_searchable_content(
        self,
        df: pd.DataFrame,
        fields: List[str] = None
    ) -> pd.DataFrame:
        """
        Combine multiple fields into a single searchable content field.
        Now includes dependency features for vocabulary matching with queries.
        
        Args:
            df: DataFrame with product data
            fields: List of field names to combine (default: name, brand, desc, semantic_tags, dependency_features)
            
        Returns:
            DataFrame with added 'searchable_content' column
        """
        if fields is None:
            fields = ["name", "brand", "desc", "semantic_tags", "dependency_features"]
        
        searchable_content = []
        for _, row in df.iterrows():
            parts = []
            for field in fields:
                if field in row and row[field]:
                    parts.append(str(row[field]))
            searchable_content.append(" ".join(parts))
        
        df["searchable_content"] = searchable_content
        return df
    
    def enrich_query_with_embeddings(
        self,
        query: str,
        max_expansions: int = 5
    ) -> str:
        """
        Enrich search query using word embeddings.
        
        Expands query terms with semantically similar words:
        - "cheap" -> "budget", "affordable", "economical"
        - "hills" -> "mountain", "climbing", "steep"
        
        Args:
            query: Original search query
            max_expansions: Maximum number of expansion terms
            
        Returns:
            Enhanced query with additional semantic terms
        """
        if not self.use_embeddings or not self.embeddings:
            return query
        
        try:
            enhanced = self.embeddings.enhance_query_with_embeddings(
                query,
                max_additions=max_expansions
            )
            return enhanced
        except Exception as e:
            logger.warning(f"Could not enhance query with embeddings: {e}")
            return query
    
    def compute_semantic_similarity(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[tuple]:
        """
        Compute semantic similarity between query and documents.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not self.use_embeddings or not self.embeddings:
            return []
        
        if not documents:
            return []
        
        cleaned_documents = []
        for doc in documents:
            if doc is None:
                cleaned_documents.append("")
            elif isinstance(doc, str):
                cleaned_documents.append(doc)
            else:
                cleaned_documents.append(str(doc))
        
        try:
            return self.embeddings.compute_similarity(
                query,
                cleaned_documents,
                top_k=top_k
            )
        except Exception as e:
            logger.warning(f"Could not compute semantic similarity: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
