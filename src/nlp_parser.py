"""
Natural Language Understanding module.
Handles brand recognition, constraint parsing, and query analysis.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.config import (
    KNOWN_BRANDS,
    AMBIGUOUS_BRAND_WORDS,
    PRICE_PATTERNS,
    DEFAULT_CUSTOM_BRANDS
)


class NLPParser:
    """Parses user queries to extract intent, brands, and constraints."""
    
    def __init__(
        self,
        custom_brands: Optional[List[str]] = None,
        known_brands: Optional[set] = None
    ):
        """
        Initialize NLP parser.
        
        Args:
            custom_brands: Additional brands to recognize beyond default list (defaults to config.DEFAULT_CUSTOM_BRANDS)
            known_brands: Override the default known brands set (defaults to config.KNOWN_BRANDS)
        """
        custom_brands = custom_brands if custom_brands is not None else DEFAULT_CUSTOM_BRANDS
        known_brands = known_brands if known_brands is not None else KNOWN_BRANDS
        
        self.brands = known_brands.copy()
        if custom_brands:
            self.brands.update(set(brand.lower() for brand in custom_brands))
    
    def extract_brands(self, query: str) -> List[str]:
        """
        Extract brand names from query using dictionary lookup.
        
        Args:
            query: User query string
            
        Returns:
            List of identified brand names
        """
        query_lower = query.lower()
        found_brands = []
        
        for brand in self.brands:
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, query_lower):
                found_brands.append(brand)
        
        return found_brands
    
    def resolve_ambiguous_terms(self, query: str) -> Dict[str, str]:
        """
        Resolve ambiguous terms that could be brands or categories.
        
        For example:
        - "Giant mountain bike" -> Giant is a brand
        - "giant screen TV" -> giant means large
        
        Args:
            query: User query string
            
        Returns:
            Dictionary mapping ambiguous terms to their resolved meaning
        """
        query_lower = query.lower()
        resolutions = {}
        #TODO [Enhancement] Currently static and very limited
        for term, meaning in AMBIGUOUS_BRAND_WORDS.items():
            if term in query_lower:
                brand_context_pattern = f"{term}\\s+(bike|bicycle|mtb|equipment|component)"
                if re.search(brand_context_pattern, query_lower):
                    resolutions[term] = "brand"
                else:
                    adjective_context_pattern = f"(a|the)\\s+{term}\\s+(bike|screen|item)"
                    if re.search(adjective_context_pattern, query_lower):
                        resolutions[term] = "adjective"
                    else:
                        resolutions[term] = meaning
        
        return resolutions
    
    def extract_price_constraints(self, query: str) -> Dict[str, float]:
        """
        Extract price constraints from natural language.
        
        Examples:
        - "under 50 dollars" -> {"max": 50}
        - "over 100 euro" -> {"min": 100}
        - "between 50 and 100" -> {"min": 50, "max": 100}
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with 'min' and/or 'max' price constraints
        """
        query_lower = query.lower()
        constraints = {}
        
        between_match = re.search(PRICE_PATTERNS["between"], query_lower)
        if between_match:
            min_price = float(between_match.group(1))
            max_price = float(between_match.group(2))
            constraints["min"] = min_price
            constraints["max"] = max_price
            return constraints
        
        under_match = re.search(PRICE_PATTERNS["under"], query_lower)
        if under_match:
            constraints["max"] = float(under_match.group(1))
        
        over_match = re.search(PRICE_PATTERNS["over"], query_lower)
        if over_match:
            constraints["min"] = float(over_match.group(1))
        
        exact_match = re.search(PRICE_PATTERNS["exact"], query_lower)
        if exact_match:
            price = float(exact_match.group(1))
            margin = price * 0.2
            constraints["min"] = price - margin
            constraints["max"] = price + margin
        
        return constraints
    
    def parse_query(self, query: str) -> Dict:
        """
        Comprehensive query parsing.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing:
            - brands: List of identified brands
            - price_constraints: Price min/max
            - weight_constraints: Weight min/max (NEW)
            - ambiguous_terms: Resolved ambiguous terms
            - clean_query: Query with brands/constraints removed
        """
        result = {
            "brands": self.extract_brands(query),
            "price_constraints": self.extract_price_constraints(query),
            "weight_constraints": self.extract_numeric_constraints(query, unit="kg"),
            "ambiguous_terms": self.resolve_ambiguous_terms(query),
            "original_query": query
        }
        
        clean = query.lower()
        
        for pattern in PRICE_PATTERNS.values():
            clean = re.sub(pattern, "", clean)
        
        weight_patterns = [
            r'(?:lighter than|under|below|less than)\s*\d+\.?\d*\s*kg',
            r'(?:heavier than|over|above|more than)\s*\d+\.?\d*\s*kg',
            r'(?:weight|weighs?)\s*(?:under|over|less than|more than)?\s*\d+\.?\d*\s*kg'
        ]
        for pattern in weight_patterns:
            clean = re.sub(pattern, "", clean)
        
        for brand in result["brands"]:
            clean = re.sub(r'\b' + re.escape(brand) + r'\b', "", clean)
        
        clean = re.sub(r'[$€£¥]', "", clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        result["clean_query"] = clean
        
        return result
    
    def extract_numeric_constraints(self, query: str, unit: str = "kg") -> Dict[str, float]:
        """
        Extract numeric constraints for attributes like weight.
        
        Example:
        - "lighter than 12kg" -> {"max": 12}
        - "weight under 15kg" -> {"max": 15}
        
        Args:
            query: User query string
            unit: Unit to look for (default: kg)
            
        Returns:
            Dictionary with min/max constraints
        """
        query_lower = query.lower()
        constraints = {}
        
        under_pattern = f"(?:lighter than|under|below|less than)\\s*(\\d+\\.?\\d*)\\s*{unit}"
        over_pattern = f"(?:heavier than|over|above|more than)\\s*(\\d+\\.?\\d*)\\s*{unit}"
        
        under_match = re.search(under_pattern, query_lower)
        if under_match:
            constraints["max"] = float(under_match.group(1))
        
        over_match = re.search(over_pattern, query_lower)
        if over_match:
            constraints["min"] = float(over_match.group(1))
        
        return constraints
