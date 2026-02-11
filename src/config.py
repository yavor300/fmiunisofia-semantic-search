"""
Configuration module for the semantic search engine.
Centralizes all configuration properties and semantic mappings.
"""

from typing import Dict, List

# ============================================================================
# NLP Configuration
# ============================================================================
LANGUAGE = "english"
MAX_FEATURES = 5000

# ============================================================================
# Elasticsearch Configuration
# ============================================================================
ELASTICSEARCH_HOST = "localhost"
ELASTICSEARCH_PORT = 9200
INDEX_NAME = "amazon_products"

# ============================================================================
# Search Configuration
# ============================================================================
DEFAULT_TOP_K = 10
MIN_SIMILARITY_SCORE = 0.01

# ============================================================================
# Text Preprocessing Configuration
# ============================================================================
DEFAULT_LANGUAGE = "english"
DEFAULT_USE_DEPENDENCY_PARSING = False

# ============================================================================
# NLP Parser Configuration
# ============================================================================
DEFAULT_CUSTOM_BRANDS = None

# ============================================================================
# Indexer Configuration
# ============================================================================
DEFAULT_MAX_FEATURES = MAX_FEATURES

# ============================================================================
# Word Embeddings Configuration
# ============================================================================
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_USE_EMBEDDING_CACHE = True

# ============================================================================
# Search Engine Configuration
# ============================================================================
DEFAULT_USE_ELASTICSEARCH = True
DEFAULT_USE_EMBEDDINGS = False
DEFAULT_HYBRID_ALPHA = 0.6

# ============================================================================
# Semantic Enrichment Configuration
# ============================================================================
DEFAULT_USE_EMBEDDINGS_FOR_ENRICHMENT = False
DEFAULT_USE_DYNAMIC_TAGS = True

# ============================================================================
# Dynamic Tag Generation Configuration
# ============================================================================
DYNAMIC_TAGS_ENABLED = False  # Set to False to use static tags
USE_EMBEDDINGS_FOR_TAGS = True  # Use word embeddings for tag generation
USE_WORDNET_FOR_TAGS = True  # Use WordNet for tag generation
CACHE_DYNAMIC_TAGS = True  # Cache generated tags for performance
MAX_SYNONYMS_PER_SEED = 3  # Maximum synonyms to generate per seed word

DEFAULT_USE_EMBEDDINGS_FOR_TAGS = USE_EMBEDDINGS_FOR_TAGS
DEFAULT_USE_WORDNET_FOR_TAGS = USE_WORDNET_FOR_TAGS
DEFAULT_CACHE_DYNAMIC_TAGS = CACHE_DYNAMIC_TAGS
DEFAULT_MAX_SYNONYMS_PER_SEED = MAX_SYNONYMS_PER_SEED

# ============================================================================
# Semantic Enrichment Rules
# ============================================================================
PRICE_THRESHOLDS = {
    "budget": (0, 30),
    "affordable": (30, 75),
    "mid_range": (75, 150),
    "premium": (150, 300),
    "luxury": (300, float('inf'))
}

# Weight extraction patterns (for bikes, equipment, etc.)
WEIGHT_THRESHOLDS = {
    "light": (0, 12),       # < 12kg
    "medium": (12, 18),     # 12-18kg
    "heavy": (18, float('inf'))  # > 18kg
}

# ============================================================================
# Seed words for dynamic tag generation
# ============================================================================
PRICE_TAG_SEEDS = {
    "budget": ["cheap", "budget", "economical", "inexpensive"],
    "affordable": ["affordable", "reasonable", "value"],
    "mid_range": ["moderate", "standard", "average", "mid-range"],
    "premium": ["premium", "expensive", "quality", "high-end"],
    "luxury": ["luxury", "expensive", "elite", "exclusive"]
}

WEIGHT_TAG_SEEDS = {
    "light": ["light", "lightweight", "portable", "compact"],
    "medium": ["medium", "standard", "average"],
    "heavy": ["heavy", "sturdy", "robust", "durable"]
}

# Seed words for category-based enrichment
CATEGORY_TAG_SEEDS = {
    "bike": ["bicycle", "cycling", "rider"],
    "mtb": ["mountain", "off-road", "trail"],
    "road": ["racing", "speed", "performance"],
    "hybrid": ["versatile", "commuting", "city"],
    "shoes": ["footwear", "sneakers"],
    "running": ["jogging", "athletic", "fitness"],
    "laptop": ["computer", "notebook", "portable"],
    "phone": ["smartphone", "mobile", "device"],
    "headphones": ["audio", "earphones", "music"],
    "camera": ["photography", "video", "shooting"]
}

# ============================================================================
# Tags for semantic enrichment (static fallback)
# ============================================================================
PRICE_TAGS = {
    "budget": "budget affordable cheap low-cost economical inexpensive",
    "affordable": "affordable reasonably-priced value",
    "mid_range": "mid-range moderate standard average",
    "premium": "premium expensive high-end quality upscale",
    "luxury": "luxury expensive high-end elite exclusive top-tier"
}

WEIGHT_TAGS = {
    "light": "light lightweight portable compact",
    "medium": "medium standard average",
    "heavy": "heavy sturdy robust durable"
}

# ============================================================================
# Known brands for brand recognition (expandable)
# ============================================================================
KNOWN_BRANDS = {
    "nike", "adidas", "puma", "reebok", "under armour",
    "new balance", "asics", "converse", "vans", "fila",
    "samsung", "apple", "sony", "lg", "dell",
    "hp", "lenovo", "asus", "acer", "microsoft",
    "shimano", "sram", "drag", "trek", "giant",
    "specialized", "cannondale", "scott", "bianchi",
    "saucony", "kishigo", "twinsluxes", "accutire"
}

# ============================================================================
# Ambiguous words that could be brands or categories
# ============================================================================
AMBIGUOUS_BRAND_WORDS = {
    "cross": "brand",  # Cross brand vs cross-country
    "giant": "brand",  # Giant brand vs giant (large)
    "trek": "brand",   # Trek brand vs trek (journey)
    "specialized": "brand"  # Specialized brand vs specialized (specific)
}

# ============================================================================
# Price extraction patterns
# ============================================================================
PRICE_PATTERNS = {
    "under": r"(?:under|below|less than|up to|max|maximum)\s*(\d+)",
    "over": r"(?:over|above|more than|at least|minimum)\s*(\d+)",
    "between": r"between\s*(\d+)\s*(?:and|to|-)\s*(\d+)",
    "exact": r"(?:around|approximately|about|roughly)\s*(\d+)"
}

# ============================================================================
# Semantic Mappings for Word Embeddings
# ============================================================================
def build_semantic_mappings() -> Dict[str, List[str]]:
    """
    Build domain-specific semantic mappings for query expansion.
    
    Returns:
        Dictionary mapping terms to their semantic expansions
    """
    return {
        # Price-related terms
        "cheap": ["budget", "affordable", "inexpensive", "economical", "low-cost"],
        "expensive": ["premium", "luxury", "high-end", "costly"],
        "affordable": ["budget", "reasonably priced", "economical", "cheap"],
        
        # Quality terms
        "quality": ["premium", "high-end", "professional", "durable"],
        "lightweight": ["light", "portable", "easy to carry", "compact"],
        "heavy-duty": ["durable", "robust", "strong", "sturdy"],
        
        # Terrain/usage for bikes
        "hills": ["mountain", "climbing", "steep", "uphill", "mtb"],
        "trails": ["off-road", "mountain", "unpaved", "rough terrain"],
        "road": ["pavement", "street", "racing", "smooth"],
        "city": ["urban", "commuting", "street", "town"],
        
        # Bike types
        "mtb": ["mountain bike", "mountain", "off-road bike"],
        "mountain bike": ["mtb", "trail bike", "off-road bike"],
        "road bike": ["racing bike", "speed bike", "road racer"],
        "hybrid": ["versatile", "all-purpose", "mixed-use"],
        
        # Family/usage
        "child": ["kid", "children", "youth", "junior"],
        "family": ["kids", "children", "group", "multiple riders"],
        
        # Performance terms
        "fast": ["speed", "quick", "rapid", "swift", "performance"],
        "comfortable": ["ergonomic", "cushioned", "soft", "padded"],
        
        # Size terms
        "small": ["compact", "mini", "petite", "little"],
        "large": ["big", "spacious", "roomy", "xl"],
        
        # Running/sports
        "running": ["jogging", "marathon", "sprint", "athletic"],
        "training": ["workout", "exercise", "fitness", "athletic"],
        
        # Technology
        "wireless": ["bluetooth", "cordless", "cable-free"],
        "portable": ["mobile", "compact", "lightweight", "travel"],
    }

SEMANTIC_MAPPINGS = build_semantic_mappings()
