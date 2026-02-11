"""
Main search engine orchestrating all components with local TFIDF or Elasticsearch if flag is enabled.
Enhanced with dependency parsing and word embeddings for advanced semantic search.
"""

import os
import re
import pandas as pd
from typing import List, Dict, Optional
import logging

from src.preprocessing import TextPreprocessor
from src.nlp_parser import NLPParser
from src.semantic_enrichment import SemanticEnricher
from src.elasticsearch_indexer import ElasticsearchIndexer
from src.config import (
    DEFAULT_TOP_K,
    DEFAULT_USE_ELASTICSEARCH,
    DEFAULT_USE_EMBEDDINGS,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_USE_DEPENDENCY_PARSING,
    DEFAULT_CUSTOM_BRANDS
)

logger = logging.getLogger(__name__)

class ProductSearchEngine:
    """
    Domain-specific product search engine with TFIDF or Elasticsearch.
    
    Combines NLP, semantic enrichment, and Elasticsearch
    for advanced product search with natural language queries.
    """
    
    def __init__(
        self,
        custom_brands: Optional[List[str]] = None,
        use_elasticsearch: bool = None,
        use_dependency_parsing: bool = None,
        use_embeddings: bool = None,
        hybrid_alpha: float = None
    ):
        """
        Initialize the search engine.
        
        Args:
            custom_brands: Additional brand names for recognition (defaults to config.DEFAULT_CUSTOM_BRANDS)
            use_elasticsearch: Whether to use Elasticsearch (defaults to config.DEFAULT_USE_ELASTICSEARCH)
            use_dependency_parsing: Enable advanced dependency parsing (defaults to config.DEFAULT_USE_DEPENDENCY_PARSING)
            use_embeddings: Enable word embeddings for semantic similarity (defaults to config.DEFAULT_USE_EMBEDDINGS)
            hybrid_alpha: Weight for TF-IDF score in hybrid mode (defaults to config.DEFAULT_HYBRID_ALPHA)
                         Score = alpha * TF-IDF + (1-alpha) * Embedding
        """
        custom_brands = custom_brands if custom_brands is not None else DEFAULT_CUSTOM_BRANDS
        use_elasticsearch = use_elasticsearch if use_elasticsearch is not None else DEFAULT_USE_ELASTICSEARCH
        use_dependency_parsing = use_dependency_parsing if use_dependency_parsing is not None else DEFAULT_USE_DEPENDENCY_PARSING
        use_embeddings = use_embeddings if use_embeddings is not None else DEFAULT_USE_EMBEDDINGS
        hybrid_alpha = hybrid_alpha if hybrid_alpha is not None else DEFAULT_HYBRID_ALPHA
        
        self.preprocessor = TextPreprocessor(use_dependency_parsing=use_dependency_parsing)
        self.nlp_parser = NLPParser(custom_brands=custom_brands)
        
        self.enricher = SemanticEnricher(
            use_embeddings=use_embeddings,
            preprocessor=self.preprocessor
        )
        
        self.use_elasticsearch = use_elasticsearch
        self.use_dependency_parsing = use_dependency_parsing
        self.use_embeddings = use_embeddings
        self.hybrid_alpha = hybrid_alpha
        
        if use_elasticsearch:
            self.indexer = ElasticsearchIndexer()
        else:
            from src.indexer import SearchIndexer
            self.indexer = SearchIndexer()
        
        self.df = None
        self.ready = False
        
        logger.info(f"Search engine initialized:")
        logger.info(f"  - Elasticsearch: {use_elasticsearch}")
        logger.info(f"  - Dependency parsing: {use_dependency_parsing}")
        logger.info(f"  - Word embeddings: {use_embeddings}")
        logger.info(f"  - Hybrid alpha: {hybrid_alpha}")
    
    def load_data(self, file_path: str, limit: Optional[int] = None) -> bool:
        """
        Load product data from CSV file.
        
        Expected columns: title, brand, description, final_price, 
                         category, availability, rating, reviews_count, etc.
        
        Args:
            file_path: Path to CSV file
            limit: Maximum number of products to load (for testing)
            
        Returns:
            True if loading was successful, False otherwise
        """
        print(f"Loading product data from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at: {file_path}")
            return False
        
        try:
            self.df = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
            
            if limit:
                self.df = self.df.head(limit)
            
            print(f"Loaded {len(self.df)} products from CSV")
            
            self._normalize_data_columns()
            
            self.df = self.df[self.df["name"].str.len() > 0]
            
            print(f"Successfully processed {len(self.df)} valid products")
            if len(self.df) > 0:
                print(f"Sample: {self.df[['name', 'brand', 'price_numeric']].head(1).values}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _normalize_data_columns(self):
        """Normalize CSV column names to internal schema."""
        
        print(f"Available CSV columns: {list(self.df.columns)[:10]}...")
        
        if "title" in self.df.columns:
            self.df["product"] = self.df["title"].fillna("").astype(str)
            self.df["name"] = self.df["product"]  # Keep 'name' for backward compatibility
        else:
            print("Warning: 'title' column not found, using empty strings")
            self.df["product"] = ""
            self.df["name"] = ""
        
        if "brand" in self.df.columns:
            self.df["brand"] = self.df["brand"].fillna("").astype(str)
        else:
            print("Warning: 'brand' column not found, using empty strings")
            self.df["brand"] = ""
        
        if "description" in self.df.columns:
            self.df["desc"] = self.df["description"].fillna("").astype(str)
        else:
            print("Warning: 'description' column not found, using empty strings")
            self.df["desc"] = ""
        
        if "final_price" in self.df.columns:
            self.df["price_raw"] = self.df["final_price"]
            self.df["price_numeric"] = self.df["final_price"].apply(self._clean_price)
        elif "price" in self.df.columns:
            self.df["price_raw"] = self.df["price"]
            self.df["price_numeric"] = self.df["price"].apply(self._clean_price)
        else:
            self.df["price_raw"] = 0
            self.df["price_numeric"] = 0
        
        self.df["price"] = self.df["price_numeric"]
        
        if "categories" in self.df.columns:
            self.df["category"] = self.df["categories"].apply(self._extract_category)
        elif "root_bs_category" in self.df.columns:
            self.df["category"] = self.df["root_bs_category"].fillna("").astype(str)
        else:
            self.df["category"] = ""
        
        if "availability" in self.df.columns:
            self.df["availability"] = self.df["availability"].fillna("Unknown").astype(str)
        else:
            self.df["availability"] = "Unknown"
        
        if "rating" in self.df.columns:
            self.df["rating"] = pd.to_numeric(self.df["rating"], errors='coerce').fillna(0)
        else:
            self.df["rating"] = 0
        
        if "reviews_count" in self.df.columns:
            self.df["reviews_count"] = pd.to_numeric(self.df["reviews_count"], errors='coerce').fillna(0).astype(int)
        else:
            self.df["reviews_count"] = 0
        
        if "image_url" in self.df.columns:
            self.df["image_url"] = self.df["image_url"].fillna("").astype(str)
        else:
            self.df["image_url"] = ""
        
        if "url" in self.df.columns:
            self.df["url"] = self.df["url"].fillna("").astype(str)
        else:
            self.df["url"] = ""
        
        if "item_weight" in self.df.columns:
            self.df["weight"] = self.df["item_weight"].apply(self._clean_weight)
        else:
            self.df["weight"] = 0
    
    def _extract_category(self, categories_str):
        """Extract main category from Amazon nested category structure."""
        if pd.isna(categories_str) or categories_str == "":
            return ""
        
        try:
            import ast
            if isinstance(categories_str, str):
                if categories_str.startswith('['):
                    categories = ast.literal_eval(categories_str)
                    if categories and len(categories) > 0:
                        for cat in reversed(categories):
                            if cat and cat not in ["All", "Products"]:
                                return cat
                        return categories[-1] if categories else ""
                else:
                    return categories_str
            return str(categories_str)
        except:
            return str(categories_str)[:100]
    
    def _clean_price(self, price_str) -> float:
        """Extract numeric price from various formats."""
        if isinstance(price_str, (int, float)):
            return float(price_str)
        
        if pd.isna(price_str):
            return 0.0
        
        found = re.findall(r"[-+]?\d*\.\d+|\d+", str(price_str))
        if found:
            return float(found[0])
        return 0.0
    
    def _clean_weight(self, weight_str) -> float:
        """Extract numeric weight in kg."""
        if isinstance(weight_str, (int, float)):
            return float(weight_str)
        
        if pd.isna(weight_str) or weight_str == "":
            return 0.0
        
        try:
            weight_str = str(weight_str).lower()

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", weight_str)
            if numbers:
                weight = float(numbers[0])
                # Convert to kg if in pounds
                if "pound" in weight_str or "lb" in weight_str:
                    weight = weight * 0.453592
                # Convert to kg if in grams
                elif "gram" in weight_str or " g" in weight_str:
                    weight = weight / 1000
                # Convert to kg if in ounces
                elif "ounce" in weight_str or "oz" in weight_str:
                    weight = weight * 0.0283495
                return weight
        except:
            pass
        
        return 0.0
    
    def build_index(self):
        """
        Build search index: preprocess, enrich, and index in Elasticsearch.
        """
        if self.df is None or self.df.empty:
            print("ERROR: No data loaded. Call load_data() first.")
            return False
        
        print("\n=== Building Search Index ===")

        print("1. Applying semantic enrichment...")
        self.df = self.enricher.enrich_dataframe(self.df)

        print("2. Creating searchable content...")
        self.df = self.enricher.create_searchable_content(self.df)
        
        if not self.use_elasticsearch:
            print("3. Preprocessing text...")
            self.df["processed_content"] = self.df["searchable_content"].apply(
                lambda x: self.preprocessor.preprocess(x)
            )
        
        if self.use_elasticsearch:
            print("4. Indexing documents in Elasticsearch...")
            self.indexer.build_index(self.df)
        else:
            print("4. Building TF-IDF index...")
            self.indexer.build_index(self.df, content_field="processed_content")
        
        self.ready = True
        print("Index built successfully!\n")
        return True
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        verbose: bool = False,
        use_query_enhancement: bool = True
    ) -> List[Dict]:
        """
        Search for products using natural language query.
        Enhanced with dependency parsing and word embeddings.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            verbose: Print detailed parsing information
            use_query_enhancement: Use dependency parsing and embeddings to enhance query
            
        Returns:
            List of product dictionaries with similarity scores
        """
        if not self.ready:
            print("ERROR: Index not built. Call build_index() first.")
            return []
        
        parsed = self.nlp_parser.parse_query(query)
        clean_query = parsed["clean_query"]
        
        if verbose:
            print("\n=== Query Analysis ===")
            print(f"Original query: {query}")
            print(f"Extracted brands: {parsed['brands']}")
            print(f"Price constraints: {parsed['price_constraints']}")
            print(f"Weight constraints: {parsed['weight_constraints']}")
            print(f"Clean query: {clean_query}")
        
        original_query = clean_query

        if use_query_enhancement:
            if self.use_dependency_parsing:
                enhanced_query = self.preprocessor.enhance_query(clean_query)
                if verbose:
                    print(f"Dependency-enhanced query: {enhanced_query}")
                clean_query = enhanced_query
            
            if self.use_embeddings:
                embedding_enhanced = self.enricher.enrich_query_with_embeddings(
                    clean_query,
                    max_expansions=5
                )
                if verbose:
                    print(f"Embedding-enhanced query: {embedding_enhanced}")
                clean_query = embedding_enhanced
        
        # Use smart preprocessing that preserves enhancement tags
        if use_query_enhancement and (self.use_dependency_parsing or self.use_embeddings):
            processed_query = self.preprocessor.preprocess_enhanced_query(clean_query, original_query)
        else:
            processed_query = self.preprocessor.preprocess(clean_query)
        
        if not processed_query:
            print("Warning: Query is empty after preprocessing")
            return []
        
        if verbose:
            print(f"Final processed query: {processed_query}")
        
        filters = {}
        
        if parsed["brands"]:
            filters["brand"] = parsed["brands"]
        
        if "min" in parsed["price_constraints"]:
            filters["price_min"] = parsed["price_constraints"]["min"]
        
        if "max" in parsed["price_constraints"]:
            filters["price_max"] = parsed["price_constraints"]["max"]
        
        if parsed["weight_constraints"]:
            if "min" in parsed["weight_constraints"]:
                filters["weight_min"] = parsed["weight_constraints"]["min"]
            
            if "max" in parsed["weight_constraints"]:
                filters["weight_max"] = parsed["weight_constraints"]["max"]
        
        results_df = self.indexer.search(
            query=processed_query,
            top_k=top_k * 2 if self.use_embeddings else top_k,
            filters=filters if filters else None
        )
        
        if results_df.empty:
            return []
        
        if self.use_embeddings and hasattr(self.enricher, 'embeddings') and self.enricher.embeddings:
            results_df = self._apply_hybrid_scoring(
                query=parsed["clean_query"],
                results_df=results_df,
                top_k=top_k,
                verbose=verbose
            )
        
        results = []
        for idx, row in results_df.head(top_k).iterrows():
            result = {
                "product": row["product"],
                "brand": row["brand"],
                "price": row["price"],
                "score": row["similarity_score"]
            }
            
            if "desc" in results_df.columns and pd.notna(row.get("desc")):
                result["description"] = str(row["desc"])
            if "category" in results_df.columns:
                result["category"] = row.get("category", "")
            if "rating" in results_df.columns:
                result["rating"] = row.get("rating", 0)
            if "reviews_count" in results_df.columns:
                result["reviews_count"] = row.get("reviews_count", 0)
            if "availability" in results_df.columns:
                result["availability"] = row.get("availability", "")
            if "image_url" in results_df.columns:
                result["image_url"] = row.get("image_url", "")
            if "url" in results_df.columns:
                result["url"] = row.get("url", "")
            
            results.append(result)
        
        if verbose and results:
            self._print_detailed_results(results_df.head(top_k), processed_query)
        
        return results
    
    def _print_detailed_results(self, results_df: pd.DataFrame, processed_query: str):
        """
        Print detailed information about search results including enrichment fields.
        
        Args:
            results_df: DataFrame with search results
            processed_query: The final processed query that was used for search
        """
        print("\n" + "=" * 80)
        print("DETAILED SEARCH RESULTS")
        print("=" * 80)
        print(f"Processed query tokens: {processed_query}")
        print(f"Total results: {len(results_df)}")
        print()
        
        for i, (idx, row) in enumerate(results_df.iterrows(), 1):
            print(f"{'─' * 80}")
            print(f"RESULT #{i} (Score: {row['similarity_score']:.4f})")
            print(f"{'─' * 80}")
            
            print(f"\nORIGINAL PRODUCT:")
            print(f"  Title: {row.get('product', 'N/A')}")
            print(f"  Brand: {row.get('brand', 'N/A')}")
            print(f"  Price: ${row.get('price', 0):.2f}")
            
            if "desc" in row and pd.notna(row.get("desc")):
                desc = str(row["desc"])
                desc_preview = desc[:150] + "..." if len(desc) > 150 else desc
                print(f"  Description: {desc_preview}")
            
            if "category" in row and row.get("category"):
                print(f"  Category: {row.get('category')}")
            
            print(f"\n SEMANTIC TAGS:")
            if "semantic_tags" in row and pd.notna(row.get("semantic_tags")):
                tags = str(row["semantic_tags"])
                if tags:
                    print(f"  {tags}")
                else:
                    print(f"  (none)")
            else:
                print(f"  (not available)")
            
            print(f"\nDEPENDENCY FEATURES:")
            if "dependency_features" in row and pd.notna(row.get("dependency_features")):
                features = str(row["dependency_features"])
                if features:
                    print(f"  {features}")
                else:
                    print(f"  (none)")
            else:
                print(f"  (not available)")
            
            print(f"\nSEARCHABLE CONTENT:")
            if "searchable_content" in row and pd.notna(row.get("searchable_content")):
                content = str(row["searchable_content"])
                content_preview = content[:3000] + "..." if len(content) > 3000 else content
                print(f"  {content_preview}")
                print(f"  (Total length: {len(content)} characters)")
            else:
                print(f"  (not available)")
            
            if "searchable_content" in row and pd.notna(row.get("searchable_content")):
                query_tokens = set(processed_query.lower().split())
                content_tokens = set(str(row["searchable_content"]).lower().split())
                matching = query_tokens.intersection(content_tokens)
                
                if matching:
                    print(f"\n✓ MATCHING TOKENS ({len(matching)}):")
                    print(f"  {', '.join(sorted(matching)[:15])}")
                    if len(matching) > 15:
                        print(f"  ... and {len(matching) - 15} more")
            
            print()  # Blank line between results
        
        print("=" * 80)
    
    def _apply_hybrid_scoring(
        self,
        query: str,
        results_df: pd.DataFrame,
        top_k: int,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Apply hybrid scoring combining TF-IDF and embedding similarity.
        
        Args:
            query: Original query
            results_df: Results from initial search
            top_k: Number of results to return
            verbose: Print scoring details
            
        Returns:
            Reranked results dataframe
        """
        try:
            if "searchable_content" in results_df.columns:
                documents = results_df["searchable_content"].tolist()
            else:
                documents = results_df["product"].tolist()
            
            if not documents:
                logger.warning("No documents to compute similarity for")
                return results_df
            
            logger.debug(f"Computing similarity for {len(documents)} documents")
            logger.debug(f"Sample document types: {[type(d).__name__ for d in documents[:3]]}")
            
            similarities = self.enricher.compute_semantic_similarity(
                query,
                documents,
                top_k=None
            )
            
            embedding_scores = {idx: score for idx, score in similarities}
            
            hybrid_scores = []
            for i, row in results_df.iterrows():
                tfidf_score = row["similarity_score"]
                embedding_score = embedding_scores.get(i, 0.0)
                
                hybrid_score = (
                    self.hybrid_alpha * tfidf_score +
                    (1 - self.hybrid_alpha) * embedding_score
                )
                hybrid_scores.append(hybrid_score)
                
                if verbose and i < 3:  # Show first 3
                    print(f"  Result {i}: TF-IDF={tfidf_score:.3f}, Embedding={embedding_score:.3f}, Hybrid={hybrid_score:.3f}")
            
            results_df["similarity_score"] = hybrid_scores
            results_df = results_df.sort_values("similarity_score", ascending=False)
            
            if verbose:
                print(f"Applied hybrid scoring (alpha={self.hybrid_alpha})")
            
        except Exception as e:
            logger.warning(f"Could not apply hybrid scoring: {e}")
        
        return results_df
    
    def get_statistics(self) -> Dict:
        """Get statistics about the indexed data."""
        if self.df is None:
            return {}
        
        stats = {
            "total_products": len(self.df),
            "unique_brands": self.df["brand"].nunique(),
            "avg_price": round(self.df["price_numeric"].mean(), 2),
            "price_range": (
                round(self.df["price_numeric"].min(), 2),
                round(self.df["price_numeric"].max(), 2)
            ),
            "avg_rating": round(self.df["rating"].mean(), 2),
            "products_with_reviews": int((self.df["reviews_count"] > 0).sum())
        }
        
        if "category" in self.df.columns:
            stats["unique_categories"] = self.df["category"].nunique()
        
        if self.use_elasticsearch:
            es_stats = self.indexer.get_index_stats()
            stats.update(es_stats)
        
        return stats
    
    def close(self):
        """Close connections and cleanup."""
        if self.use_elasticsearch and hasattr(self.indexer, 'close'):
            self.indexer.close()
