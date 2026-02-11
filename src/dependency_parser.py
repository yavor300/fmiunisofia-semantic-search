"""
Advanced dependency parsing module using spaCy.
Extracts relationships between words to better understand complex queries.
"""

import spacy
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class DependencyParser:
    """
    Uses spaCy for advanced dependency parsing to understand grammatical relationships.
    
    Examples:
        - "bike with child seat" -> extracts (bike, with, child_seat) relationship
        - "mountain bike for hills" -> extracts (mountain_bike, for, hills) relationship
        - "lightweight carbon frame" -> extracts compound noun phrase
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the dependency parser with a spaCy model.
        
        Args:
            model_name: spaCy model to use (en_core_web_sm, en_core_web_md, en_core_web_lg)
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)
            logger.info(f"Downloaded and loaded spaCy model: {model_name}")
        
        self.relationship_preps = {
            "with", "for", "on", "in", "at", "by", "about", 
            "under", "over", "through", "without"
        }
        
        self.compound_deps = {"compound", "amod", "nmod"}
    
    def parse(self, text: str) -> spacy.tokens.Doc:
        """
        Parse text and return spaCy Doc object with dependency information
        
        Args:
            text: Input text to parse
            
        Returns:
            spaCy Doc object with parsed dependencies
        """
        return self.nlp(text)
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases using dependency parsing
        
        Args:
            text: Input text
            
        Returns:
            List of noun phrases
        """
        doc = self.parse(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if len(phrase) > 1:
                noun_phrases.append(phrase)
        
        return noun_phrases
    
    def extract_compound_terms(self, text: str) -> List[str]:
        """
        Extract compound terms using dependency relations
        
        Examples:
            - "carbon fiber frame" -> ["carbon fiber", "carbon fiber frame"]
            - "mountain bike trail" -> ["mountain bike", "bike trail"]
        
        Args:
            text: Input text
            
        Returns:
            List of compound terms
        """
        doc = self.parse(text)
        compounds = []
        
        for token in doc:
            if token.dep_ in self.compound_deps:
                compound_parts = [token.text]
                head = token.head
                compound_parts.append(head.text)

                for child in head.children:
                    if child.dep_ in self.compound_deps and child != token:
                        compound_parts.insert(0, child.text)
                
                compound = " ".join(compound_parts)
                if compound not in compounds:
                    compounds.append(compound)
        
        return compounds
    
    def extract_prepositional_relationships(
        self, 
        text: str
    ) -> List[Dict[str, str]]:
        """
        Extract relationships indicated by prepositions.
        
        Example:
            "bike with child seat" -> 
            [{
                "head": "bike",
                "prep": "with", 
                "object": "child seat",
                "relation": "bike_with_child_seat"
            }]
        
        Args:
            text: Input text
            
        Returns:
            List of relationship dictionaries
        """
        doc = self.parse(text)
        relationships = []
        
        for token in doc:
            if token.dep_ == "prep" and token.text.lower() in self.relationship_preps:
                prep = token.text.lower()
                head = token.head.text
                
                obj_tokens = []
                for child in token.children:
                    if child.dep_ == "pobj":
                        # Get the full noun phrase for the object
                        obj_tokens = self._get_subtree_text(child)
                        break
                
                if obj_tokens:
                    obj = " ".join(obj_tokens)
                    relationships.append({
                        "head": head,
                        "prep": prep,
                        "object": obj,
                        "relation": f"{head}_{prep}_{obj.replace(' ', '_')}"
                    })
        
        return relationships
    
    def _get_subtree_text(self, token) -> List[str]:
        """
        Get all text in the subtree of a token (including modifiers).
        
        Args:
            token: spaCy token
            
        Returns:
            List of words in the subtree
        """
        subtree = []
        for t in token.subtree:
            if not t.is_punct:
                subtree.append(t.text)
        return subtree
    
    def extract_entity_relationships(self, text: str) -> List[Dict[str, str]]:
        """
        Extract relationships between named entities and other words.
        
        Args:
            text: Input text
            
        Returns:
            List of entity relationships
        """
        doc = self.parse(text)
        relationships = []
        
        for ent in doc.ents:
            # Find verbs or prepositions connected to this entity
            entity_root = None
            for token in ent:
                if token.dep_ in ["nsubj", "dobj", "pobj"]:
                    entity_root = token.head
                    break
            
            if entity_root:
                relationships.append({
                    "entity": ent.text,
                    "entity_type": ent.label_,
                    "relation": entity_root.dep_,
                    "related_to": entity_root.text
                })
        
        return relationships
    
    def get_key_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text by identifying important noun phrases and entities.
        
        Args:
            text: Input text
            
        Returns:
            List of key concepts (deduplicated)
        """
        doc = self.parse(text)
        concepts = set()
        
        for ent in doc.ents:
            concepts.add(ent.text.lower())
        
        for token in doc:
            if token.dep_ in ["nsubj", "dobj", "pobj"] and token.pos_ == "NOUN":
                # Get the full noun phrase
                phrase_tokens = [token.text]
                for child in token.children:
                    if child.dep_ in ["amod", "compound"]:
                        phrase_tokens.insert(0, child.text)
                
                phrase = " ".join(phrase_tokens).lower()
                concepts.add(phrase)
        
        return list(concepts)
    
    def analyze_query_structure(self, query: str) -> Dict:
        """
        Comprehensive query analysis using dependency parsing.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with structured query information
        """
        doc = self.parse(query)
        
        analysis = {
            "original_query": query,
            "noun_phrases": self.extract_noun_phrases(query),
            "compound_terms": self.extract_compound_terms(query),
            "relationships": self.extract_prepositional_relationships(query),
            "key_concepts": self.get_key_concepts(query),
            "entities": [{"text": ent.text, "type": ent.label_} for ent in doc.ents],
            "main_action": None
        }
        
        # Find main verb (if any)
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
                analysis["main_action"] = token.lemma_
                break
        
        return analysis
    
    def enhance_query_for_search(self, query: str) -> str:
        """
        Enhance query by expanding compound terms and relationships.
        
        Example:
            "bike with child seat" -> "bike with child seat bike_with_child_seat"
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query with additional terms
        """
        analysis = self.analyze_query_structure(query)

        enhanced_terms = [query]
        seen_terms = {query.strip().lower()}
        seen_tokens = set(query.lower().split())

        def add_term(term: str, require_new_tokens: bool = True):
            """Add a term only if it contributes new information."""
            if not term:
                return

            normalized = term.strip().lower()
            if not normalized or normalized in seen_terms:
                return

            tokens = normalized.split()
            has_new_tokens = any(token not in seen_tokens for token in tokens)

            if require_new_tokens and not has_new_tokens:
                return

            enhanced_terms.append(term.strip())
            seen_terms.add(normalized)
            seen_tokens.update(tokens)

        for rel in analysis["relationships"]:
            add_term(rel.get("relation", ""), require_new_tokens=False)

        for term in analysis["compound_terms"]:
            underscore_term = term.replace(" ", "_")
            add_term(underscore_term, require_new_tokens=True)

        for term in analysis["key_concepts"]:
            underscore_term = term.replace(" ", "_")
            add_term(underscore_term, require_new_tokens=True)

        return " ".join(enhanced_terms)


if __name__ == "__main__":
    parser = DependencyParser()
    
    test_queries = [
        "bike with child seat",
        "bike for child",
        "lightweight mountain bike for hills",
        "carbon fiber frame road bike",
        "running shoes for marathon training"
    ]
    
    print("=" * 80)
    print("DEPENDENCY PARSING EXAMPLES")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        analysis = parser.analyze_query_structure(query)
        
        print(f"Noun Phrases: {analysis['noun_phrases']}")
        print(f"Compound Terms: {analysis['compound_terms']}")
        print(f"Key Concepts: {analysis['key_concepts']}")
        
        if analysis['relationships']:
            print("Relationships:")
            for rel in analysis['relationships']:
                print(f"  - {rel['head']} {rel['prep']} {rel['object']}")
        
        enhanced = parser.enhance_query_for_search(query)
        print(f"Enhanced Query: {enhanced}")
