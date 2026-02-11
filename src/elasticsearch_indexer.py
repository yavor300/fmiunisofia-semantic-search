"""
Elasticsearch indexer module for document indexing and search.
Replaces the local TF-IDF indexer with Elasticsearch running in Docker.
Enhanced with dense vector support for semantic embeddings.
"""

import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, NotFoundError
import logging

from src.config import (
    ELASTICSEARCH_HOST,
    ELASTICSEARCH_PORT,
    INDEX_NAME,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_SCORE
)

logger = logging.getLogger(__name__)


class ElasticsearchIndexer:
    """
    Elasticsearch-based indexer for product search.
    Each product is treated as a document in the corpus.
    """
    
    def __init__(
        self,
        host: str = ELASTICSEARCH_HOST,
        port: int = ELASTICSEARCH_PORT,
        index_name: str = INDEX_NAME,
        use_vectors: bool = False
    ):
        """
        Initialize Elasticsearch connection.
        
        Args:
            host: Elasticsearch host
            port: Elasticsearch port
            index_name: Name of the index to create/use
            use_vectors: Enable dense vector support for semantic search
        """
        self.host = host
        self.port = port
        self.index_name = index_name
        self.es = None
        self.df = None
        self.use_vectors = use_vectors
        self.embeddings = None
        
        if use_vectors:
            try:
                from src.word_embeddings import SemanticEmbeddings
                self.embeddings = SemanticEmbeddings()
                logger.info("Dense vector support enabled")
            except Exception as e:
                logger.warning(f"Could not initialize embeddings: {e}")
                self.use_vectors = False
        
        self._connect()
    
    def _connect(self, max_retries: int = 5, retry_delay: int = 2):
        """
        Connect to Elasticsearch with retry logic.
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        for attempt in range(max_retries):
            try:
                self.es = Elasticsearch(
                    [f"http://{self.host}:{self.port}"],
                    request_timeout=30
                )
                
                if self.es.ping():
                    print(f"Connected to Elasticsearch at {self.host}:{self.port}")
                    return
                else:
                    raise ConnectionError("Elasticsearch ping failed")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Connection attempt {attempt + 1} failed. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to Elasticsearch after {max_retries} attempts. "
                        f"Make sure Elasticsearch is running: docker compose up -d"
                    ) from e
    
    def create_index(self):
        """
        Create Elasticsearch index with custom mappings and settings.
        Optimized for product search with semantic fields.
        """
        if self.es.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        index_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "product_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "english_stop",
                                "english_stemmer",
                                "asciifolding"
                            ]
                        }
                    },
                    "filter": {
                        "english_stop": {
                            "type": "stop",
                            "stopwords": "_english_"
                        },
                        "english_stemmer": {
                            "type": "stemmer",
                            "language": "english"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "brand": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "price": {"type": "float"},
                    "weight": {"type": "float"},
                    "category": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "semantic_tags": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "searchable_content": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "availability": {"type": "keyword"},
                    "rating": {"type": "float"},
                    "reviews_count": {"type": "integer"},
                    "image_url": {"type": "keyword"},
                    "url": {"type": "keyword"}
                }
            }
        }
        
        if self.use_vectors and self.embeddings:
            embedding_dim = self.embeddings.model.get_sentence_embedding_dimension()
            index_body["mappings"]["properties"]["content_vector"] = {
                "type": "dense_vector",
                "dims": embedding_dim,
                "index": True,
                "similarity": "cosine"
            }
            logger.info(f"Added dense_vector field with dimension: {embedding_dim}")
        
        self.es.indices.create(index=self.index_name, body=index_body)
        print(f"Created index: {self.index_name}")
    
    def build_index(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Index all products from dataframe into Elasticsearch.
        
        Args:
            df: DataFrame with product data (already preprocessed and enriched)
            batch_size: Number of documents to index per batch
        """
        if df.empty:
            raise ValueError("Cannot build index on empty dataframe")
        
        self.df = df
        
        self.create_index()
        
        print(f"Indexing {len(df)} products into Elasticsearch...")
        
        print(f"Available columns in dataframe: {list(df.columns)}")
        
        content_vectors = None
        if self.use_vectors and self.embeddings:
            print("Computing embeddings for all documents...")
            texts_to_embed = (
                df["searchable_content"].tolist()
                if "searchable_content" in df.columns
                else df["name"].tolist()
            )
            content_vectors = self.embeddings.encode(texts_to_embed, normalize=True)
            print(f"Computed {len(content_vectors)} embeddings")
        
        actions = []
        failed_docs = []
        
        for idx, row in df.iterrows():
            try:
                price_val = row.get("price_numeric", 0)
                if pd.isna(price_val):
                    price_val = 0.0
                else:
                    price_val = float(price_val)
                
                rating_val = row.get("rating", 0)
                if pd.isna(rating_val):
                    rating_val = 0.0
                else:
                    rating_val = float(rating_val)
                
                reviews_val = row.get("reviews_count", 0)
                if pd.isna(reviews_val):
                    reviews_val = 0
                else:
                    reviews_val = int(reviews_val)
                
                weight_val = row.get("weight", 0)
                if pd.isna(weight_val):
                    weight_val = 0.0
                else:
                    weight_val = float(weight_val)
                
                source_doc = {
                    "product_id": str(idx),
                    "title": str(row.get("name", ""))[:500],
                    "brand": str(row.get("brand", ""))[:200],
                    "description": str(row.get("desc", ""))[:3000],
                    "price": price_val,
                    "weight": weight_val,
                    "category": str(row.get("category", ""))[:200],
                    "semantic_tags": str(row.get("semantic_tags", ""))[:500],
                    "searchable_content": str(row.get("searchable_content", ""))[:3000],
                    "availability": str(row.get("availability", "Unknown"))[:100],
                    "rating": rating_val,
                    "reviews_count": reviews_val,
                    "image_url": str(row.get("image_url", ""))[:500],
                    "url": str(row.get("url", ""))[:500]
                }
                
                if content_vectors is not None:
                    pos = df.index.get_loc(idx)
                    source_doc["content_vector"] = content_vectors[pos].tolist()
                
                doc = {
                    "_index": self.index_name,
                    "_id": str(idx),
                    "_source": source_doc
                }
                actions.append(doc)
            except Exception as e:
                failed_docs.append({
                    "index": idx,
                    "error": str(e),
                    "title": row.get("name", "N/A")
                })
                if len(failed_docs) <= 5:
                    print(f"Warning: Failed to prepare document {idx}: {e}")
        
        if failed_docs:
            print(f"Failed to prepare {len(failed_docs)} documents before indexing")
        
        if not actions:
            print("No valid documents to index!")
            return
        
        print(f"Prepared {len(actions)} documents for indexing...")
        
        try:
            success_count = 0
            error_count = 0
            
            for ok, response in helpers.streaming_bulk(
                self.es,
                actions,
                chunk_size=batch_size,
                raise_on_error=False
            ):
                if ok:
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 5:
                        print(f"Indexing error: {response}")
            
            self.es.indices.refresh(index=self.index_name)
            
            print(f"Indexed {success_count} documents successfully")
            if error_count > 0:
                print(f"Failed to index {error_count} documents")
            
            stats = self.es.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            print(f"Total documents in index: {doc_count}")
            
        except Exception as e:
            print(f"Bulk indexing error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_SIMILARITY_SCORE,
        filters: dict = None,
        use_vector_search: bool = None
    ) -> pd.DataFrame:
        """
        Search for products using Elasticsearch with optional vector search.
        
        Args:
            query: Preprocessed search query
            top_k: Number of results to return
            min_score: Minimum relevance score threshold
            filters: Dictionary of filters (brand, price_min, price_max, weight_min, weight_max, etc.)
            use_vector_search: Override vector search setting (defaults to self.use_vectors)
            
        Returns:
            DataFrame with search results including relevance scores
        """
        if not query:
            return pd.DataFrame()
        
        if use_vector_search is None:
            use_vector_search = self.use_vectors
        
        if use_vector_search and self.embeddings:
            return self._hybrid_search(query, top_k, min_score, filters)
        
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "searchable_content^3",
                        "title^2",
                        "brand^1.5",
                        "description",
                        "semantic_tags^2",
                        "category"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        ]
        
        filter_clauses = []
        
        if filters:
            if "brand" in filters:
                brands = filters["brand"]
                if isinstance(brands, str):
                    brands = [brands]
                filter_clauses.append({
                    "terms": {"brand.keyword": [b.title() for b in brands]}
                })
            
            # Price filters
            price_range = {}
            if "price_min" in filters:
                price_range["gte"] = filters["price_min"]
            if "price_max" in filters:
                price_range["lte"] = filters["price_max"]
            
            if price_range:
                filter_clauses.append({
                    "range": {"price": price_range}
                })
            
            # Weight filters (NEW)
            weight_range = {}
            if "weight_min" in filters:
                weight_range["gte"] = filters["weight_min"]
            if "weight_max" in filters:
                weight_range["lte"] = filters["weight_max"]
            
            if weight_range:
                filter_clauses.append({
                    "range": {"weight": weight_range}
                })
            
            if "category" in filters:
                categories = filters["category"]
                if isinstance(categories, str):
                    categories = [categories]
                filter_clauses.append({
                    "terms": {"category.keyword": categories}
                })
            
            if "availability" in filters:
                filter_clauses.append({
                    "term": {"availability": filters["availability"]}
                })
        
        es_query = {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
        
        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": es_query,
                    "size": top_k,
                    "min_score": min_score,
                    "_source": [
                        "title", "brand", "price", "weight", "description", "category",
                        "rating", "reviews_count", "availability",
                        "image_url", "url"
                    ]
                }
            )
        except NotFoundError:
            print(f"Index '{self.index_name}' not found. Please build the index first.")
            return pd.DataFrame()
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            result = {
                "product": source.get("title", ""),
                "brand": source.get("brand", ""),
                "price": source.get("price", 0),
                "similarity_score": hit['_score'] / 100,  # Normalize score
                "desc": source.get("description", ""),
                "category": source.get("category", ""),
                "rating": source.get("rating", 0),
                "reviews_count": source.get("reviews_count", 0),
                "availability": source.get("availability", ""),
                "image_url": source.get("image_url", ""),
                "url": source.get("url", "")
            }
            results.append(result)
        
        if not results:
            return pd.DataFrame()
        
        if self.df is not None and "searchable_content" in self.df.columns:
            results_df = pd.DataFrame(results)
            if not results_df.empty:
                for idx, row in results_df.iterrows():
                    matches = self.df[self.df["name"] == row["product"]]
                    if not matches.empty:
                        results_df.at[idx, "searchable_content"] = matches.iloc[0].get("searchable_content", "")
                return results_df
        
        return pd.DataFrame(results)
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        filters: dict = None
    ) -> pd.DataFrame:
        """
        Perform hybrid search combining text search and vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold
            filters: Search filters
            
        Returns:
            DataFrame with search results
        """
        try:
            query_vector = self.embeddings.get_query_embedding(query)
            
            filter_clauses = []
            if filters:
                if "brand" in filters:
                    brands = filters["brand"]
                    if isinstance(brands, str):
                        brands = [brands]
                    filter_clauses.append({
                        "terms": {"brand.keyword": [b.title() for b in brands]}
                    })
                
                # Price filters
                price_range = {}
                if "price_min" in filters:
                    price_range["gte"] = filters["price_min"]
                if "price_max" in filters:
                    price_range["lte"] = filters["price_max"]
                
                if price_range:
                    filter_clauses.append({
                        "range": {"price": price_range}
                    })
                
                # Weight filters (NEW)
                weight_range = {}
                if "weight_min" in filters:
                    weight_range["gte"] = filters["weight_min"]
                if "weight_max" in filters:
                    weight_range["lte"] = filters["weight_max"]
                
                if weight_range:
                    filter_clauses.append({
                        "range": {"weight": weight_range}
                    })
            
            es_query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": [
                                            "searchable_content^3",
                                            "title^2",
                                            "brand^1.5",
                                            "description",
                                            "semantic_tags^2",
                                            "category"
                                        ],
                                        "type": "best_fields",
                                        "fuzziness": "AUTO"
                                    }
                                }
                            ],
                            "filter": filter_clauses
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_vector') + 1.0",
                        "params": {"query_vector": query_vector.tolist()}
                    }
                }
            }
            
            response = self.es.search(
                index=self.index_name,
                body={
                    "query": es_query,
                    "size": top_k,
                    "_source": [
                        "title", "brand", "price", "weight", "description", "category",
                        "rating", "reviews_count", "availability",
                        "image_url", "url", "searchable_content"
                    ]
                }
            )
            
            if not response["hits"]["hits"]:
                return pd.DataFrame()
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append({
                    "product": source.get("title", ""),
                    "brand": source.get("brand", ""),
                    "price": source.get("price", 0),
                    "desc": source.get("description", ""),
                    "category": source.get("category", ""),
                    "rating": source.get("rating", 0),
                    "reviews_count": source.get("reviews_count", 0),
                    "availability": source.get("availability", ""),
                    "image_url": source.get("image_url", ""),
                    "url": source.get("url", ""),
                    "searchable_content": source.get("searchable_content", ""),
                    "similarity_score": hit["_score"] / 100  # Normalize
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            import traceback
            traceback.print_exc()
            return self.search(query, top_k, min_score, filters, use_vector_search=False)
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        try:
            stats = self.es.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            size_bytes = stats['indices'][self.index_name]['total']['store']['size_in_bytes']
            
            return {
                "total_documents": doc_count,
                "index_size_mb": round(size_bytes / (1024 * 1024), 2),
                "index_name": self.index_name
            }
        except NotFoundError:
            return {
                "total_documents": 0,
                "index_size_mb": 0,
                "index_name": self.index_name,
                "error": "Index not found"
            }
    
    def delete_index(self):
        """Delete the index."""
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            print(f"Deleted index: {self.index_name}")
    
    def close(self):
        """Close Elasticsearch connection."""
        if self.es:
            self.es.close()
            print("Elasticsearch connection closed")
