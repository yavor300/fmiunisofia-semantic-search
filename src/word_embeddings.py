"""
Word embeddings module for semantic similarity and query expansion.
Uses Sentence Transformers and Word2Vec for deep semantic understanding.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
import logging
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_USE_EMBEDDING_CACHE,
    SEMANTIC_MAPPINGS
)

logger = logging.getLogger(__name__)


class SemanticEmbeddings:
    """
    Handles semantic embeddings for understanding word/phrase meanings.
    
    Features:
        - Sentence-level embeddings using transformers
        - Semantic similarity computation
        - Query expansion with synonyms
        - Context-aware term expansion
    """
    
    def __init__(
        self, 
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        use_cache: bool = DEFAULT_USE_EMBEDDING_CACHE,
        semantic_mappings: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the embeddings model.
        
        Args:
            model_name: Sentence transformer model name
                       - all-MiniLM-L6-v2: Fast, good quality (recommended)
                       - all-mpnet-base-v2: Higher quality, slower
                       - paraphrase-multilingual: For multi-language support
            use_cache: Whether to cache embeddings for reuse
            semantic_mappings: Optional custom semantic mappings (defaults to config.SEMANTIC_MAPPINGS)
        """
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.use_cache = use_cache
        self.embedding_cache = {} if use_cache else None
        
        self.semantic_mappings = semantic_mappings if semantic_mappings is not None else SEMANTIC_MAPPINGS
        
        self._reverse_mappings = self._build_reverse_index()
        
        logger.info(f"Loaded embedding model with dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def _build_reverse_index(self) -> Dict[str, List[str]]:
        """
        Build reverse index for bidirectional semantic lookups.
        
        Maps each value to its key(s), enabling expansion even when the
        query term is a synonym rather than the primary term.
        
        Example:
            Input mappings: {"cheap": ["budget", "affordable"], "expensive": ["premium"]}
            Reverse index: {
                "budget": ["cheap"],
                "affordable": ["cheap"],
                "premium": ["expensive"]
            }
        
        Returns:
            Dictionary mapping value terms to their primary keys
        """
        reverse_index = {}
        
        for primary_term, synonyms in self.semantic_mappings.items():
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                if synonym_lower not in reverse_index:
                    reverse_index[synonym_lower] = []
                reverse_index[synonym_lower].append(primary_term)
        
        return reverse_index
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        cleaned_texts = []
        for text in texts:
            if text is None:
                cleaned_texts.append("")
            elif isinstance(text, str):
                cleaned_texts.append(text.strip() if text.strip() else "empty")
            else:
                cleaned_texts.append(str(text).strip() if str(text).strip() else "empty")
        
        texts = cleaned_texts
        
        if self.use_cache:
            cached_embeddings = []
            cached_indices = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    cached_embeddings.append(self.embedding_cache[text])
                    cached_indices.append(i)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if uncached_texts:
                new_embeddings = self.model.encode(
                    uncached_texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False
                )

                # Update cache with new embeddings
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self.embedding_cache[text] = embedding
                
                all_embeddings = [None] * len(texts)
                
                for idx, emb in zip(cached_indices, cached_embeddings):
                    all_embeddings[idx] = emb
                
                for idx, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[idx] = emb
                
                return np.array(all_embeddings)
            else:
                # All embeddings were cached
                return np.array(cached_embeddings)
        else:
            return self.model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
    
    def compute_similarity(
        self, 
        query: str, 
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Compute semantic similarity between query and documents.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Return only top K most similar (None for all)
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        if not documents:
            return []
        
        cleaned_docs = []
        doc_indices = []
        
        for i, doc in enumerate(documents):
            if doc is None:
                doc_str = ""
            elif isinstance(doc, str):
                doc_str = doc.strip()
            else:
                doc_str = str(doc).strip()
            
            if doc_str:
                cleaned_docs.append(doc_str)
                doc_indices.append(i)
        
        if not cleaned_docs:
            return []
        
        try:
            query_embedding = self.encode([query])[0]
            doc_embeddings = self.encode(cleaned_docs)
            
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                doc_embeddings
            )[0]
            
            results = [
                (doc_indices[i], float(score)) 
                for i, score in enumerate(similarities)
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            
            if top_k:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return [(i, 0.0) for i in doc_indices]
    
    def expand_query(
        self,
        query: str,
        max_expansions: int = 3,
        similarity_threshold: float = 0.6
    ) -> List[str]:
        """
        Expand query with semantically similar terms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansion terms per keyword
            similarity_threshold: Minimum similarity for expansion
            
        Returns:
            List of expansion terms
        """
        expanded_terms = []
        query_lower = query.lower()
        
        words = query_lower.split()
        
        for word in words:
            # Try forward lookup: word is a primary term (e.g., "cheap")
            if word in self.semantic_mappings:
                expansions = self.semantic_mappings[word][:max_expansions]
                expanded_terms.extend(expansions)
            
            # Try reverse lookup: word is a synonym (e.g., "budget" → find "cheap")
            elif word in self._reverse_mappings:

                primary_terms = self._reverse_mappings[word]
                
                for primary_term in primary_terms:
                    if primary_term in self.semantic_mappings:

                        related_terms = self.semantic_mappings[primary_term][:max_expansions]
                        
                        for term in related_terms:
                            if term.lower() != word:
                                expanded_terms.append(term)
                        
                        break
        
        return expanded_terms
    
    def find_similar_terms(
        self,
        term: str,
        candidate_terms: List[str],
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find semantically similar terms from a list of candidates.
        
        Args:
            term: Query term
            candidate_terms: List of candidate terms to compare
            top_k: Number of similar terms to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (term, similarity) tuples
        """
        if not candidate_terms:
            return []
        
        term_embedding = self.encode([term])[0]
        candidate_embeddings = self.encode(candidate_terms)
        
        similarities = cosine_similarity(
            term_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        results = [
            (candidate_terms[i], float(score))
            for i, score in enumerate(similarities)
            if score >= similarity_threshold
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def enhance_query_with_embeddings(
        self,
        query: str,
        vocabulary: Optional[List[str]] = None,
        max_additions: int = 5
    ) -> str:
        """
        Enhance query by adding semantically similar terms.
        
        Args:
            query: Original query
            vocabulary: Optional vocabulary to search for similar terms
            max_additions: Maximum terms to add
            
        Returns:
            Enhanced query string
        """
        enhanced_terms = [query]
        
        # Add domain-specific expansions
        expansions = self.expand_query(query, max_expansions=max_additions)
        #TODO: [Might] Extend the count so that more other words are covered
        enhanced_terms.extend(expansions[:max_additions])
        
        if vocabulary:
            words = query.lower().split()
            for word in words:
                similar = self.find_similar_terms(
                    word,
                    vocabulary,
                    top_k=2,
                    similarity_threshold=0.7
                )
                for term, score in similar:
                    if term.lower() not in query.lower():
                        enhanced_terms.append(term)
        
        unique_terms = list(dict.fromkeys(enhanced_terms))
        return " ".join(unique_terms[:max_additions + 1])
    
    def compute_hybrid_score(
        self,
        query: str,
        document: str,
        tfidf_score: float,
        alpha: float = 0.5
    ) -> float:
        """
        Compute hybrid score combining TF-IDF and embedding similarity.
        
        Args:
            query: Search query
            document: Document text
            tfidf_score: TF-IDF similarity score
            alpha: Weight for TF-IDF (1-alpha for embeddings)
            
        Returns:
            Hybrid similarity score
        """
        query_emb = self.encode([query])[0]
        doc_emb = self.encode([document])[0]
        
        embedding_score = float(cosine_similarity(
            query_emb.reshape(1, -1),
            doc_emb.reshape(1, -1)
        )[0][0])
        
        hybrid_score = alpha * tfidf_score + (1 - alpha) * embedding_score
        
        return hybrid_score
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get embedding vector for a query.
        
        Args:
            query: Search query
            
        Returns:
            Embedding vector
        """
        return self.encode([query])[0]
    
    def batch_compute_similarities(
        self,
        queries: List[str],
        documents: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix for multiple queries and documents.
        
        Args:
            queries: List of queries
            documents: List of documents
            
        Returns:
            Similarity matrix of shape (len(queries), len(documents))
        """
        query_embeddings = self.encode(queries)
        doc_embeddings = self.encode(documents)
        
        return cosine_similarity(query_embeddings, doc_embeddings)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.use_cache:
            self.embedding_cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the embedding cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.use_cache:
            return {"cache_enabled": False}
        
        return {
            "cache_enabled": True,
            "cached_items": len(self.embedding_cache),
            "model": self.model_name
        }


if __name__ == "__main__":
    embeddings = SemanticEmbeddings()
    
    test_queries = [
        "cheap bike for hills",
        "affordable mountain bike",
        "lightweight road bike for racing"
    ]
    
    test_docs = [
        "Budget mountain bike perfect for hill climbing and steep terrain",
        "Professional racing road bike with carbon fiber frame",
        "Affordable hybrid bike for city commuting",
        "Premium MTB for off-road trails and mountains"
    ]
    
    print("=" * 80)
    print("WORD EMBEDDINGS & SEMANTIC SIMILARITY EXAMPLES")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        expansions = embeddings.expand_query(query)
        print(f"Expanded terms: {expansions}")
        
        enhanced = embeddings.enhance_query_with_embeddings(query)
        print(f"Enhanced query: {enhanced}")
        
        similarities = embeddings.compute_similarity(query, test_docs, top_k=3)
        print("\nTop matching documents:")
        for idx, score in similarities:
            print(f"  Score {score:.3f}: {test_docs[idx][:60]}...")
    
    print("\n" + "=" * 80)
    print("Cache Statistics:")
    print(embeddings.get_cache_stats())
