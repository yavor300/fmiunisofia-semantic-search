# Domain-Specific Intelligent Product Search Engine

## 1. Project Overview
This project implements an advanced Information Retrieval (IR) system designed for domain-specific e-commerce environments. Unlike traditional search engines that rely solely on keyword matching or static category filters, this system utilizes Natural Language Processing (NLP) and a Vector Space Model to understand user intent and product semantics.

The primary motivation behind this solution is to assist customers who may lack expert knowledge about complex product specifications (e.g., bicycle components, technical equipment). By allowing users to describe their needs in free text—similar to a prompt for an AI model—the system bridges the gap between vague user queries and structured product catalogs.

**The Goal:** To build a system that allows users to describe their needs in free text—similar to a prompt for an AI model—and retrieves the most relevant products using a hybrid approach of classical Information Retrieval (IR) and Natural Language Processing (NLP).

## 2. Key Features

### 2.1. Hybrid Search Mechanism
The system moves beyond simple keyword matching by combining:
* **Inverted Indexing:** For fast retrieval of documents containing specific terms.
* **Vector Space Model:** Representing products and queries as vectors in a multi-dimensional space to find semantic matches.
* **Parametric Filtering:** "Hard" filters for constraints like Price and Brand are applied to narrow down the search space before vector scoring.

### 2.2. Semantic Enrichment
A rule-based NLP module analyzes raw product data before indexing to inject "hidden" semantic tags that are not present in the original description.
* **Logic Example:** If a product price is `< 600`, the tag "budget" is added; if weight is `< 12kg`, the tag "light" is added.
* **Benefit:** Users can search for "light budget bike" and find relevant results even if those exact words are missing from the manufacturer's text.

### 2.3. Natural Language Understanding (NLP)
The system employs NLP techniques to parse user intent:
* **Brand Recognition:** Identifies brands (e.g., "Drag", "Shimano") using dictionary lookups or deterministic automata (DAWG).
* **POS Tagging & Ambiguity Resolution:** Distinguishes between brands and common words. For instance, determining if "Cross" refers to the bicycle brand or the "Cross-country" category based on context.
* **Constraint Parsing:** Extracts logical constraints from natural language, such as converting "not above 600 euro" into a mathematical filter `price <= 600`.

## 3. System Architecture

The solution follows a modular architecture as defined in the project plan:

1.  **Data Ingestion:**
    * Raw data includes Title, Description, Price, and Specifications.
    * Data is treated as a corpus of "documents".

2.  **Preprocessing Pipeline (NLP):**
    * **Tokenization:** Splitting text into constituent parts.
    * **Stop-word Removal:** Eliminating non-significant words.
    * **Stemming:** Reducing words to their root form (e.g., "running" -> "run").

3.  **Indexing Core:**
    * An **Inverted Index** associates terms with documents using Elasticseach running in Docker environment.
    * **TF-IDF (Term Frequency-Inverse Document Frequency)** weighting is used to calculate the importance of each term within a product description.

4.  **Search & Ranking:**
    * The user query is vectorized.
    * **Cosine Similarity** is calculated between the query vector and product vectors to rank results based on relevance.

5. **Dependency Parsing:** **IMPLEMENTED** - Uses spaCy to understand relationships between words (e.g., "bike with child seat" vs "bike for child"). Extracts compound terms, prepositional relationships, and key concepts from queries.

6. **Word Embeddings:** **IMPLEMENTED** - Integration of Sentence Transformers to capture deep semantic synonyms (e.g., associating "hills" with "mountain bike/MTB"). Includes:
   - Query expansion with semantic terms
   - Hybrid scoring (TF-IDF + embeddings)
   - Dense vector search in Elasticsearch

## 4. Technical Implementation

The project is built using **Python** and leverages the following logic:

* **Search Algorithm:** Uses Cosine Similarity formula: $sim(q,d) = \frac{q \cdot d}{|q| \cdot |d|}$.
* **Logic Extraction:** Regular Expressions (RegEx) and dependency logic are used to handle price limits (e.g., "under 50").
* **Libraries:**
    * `pandas` for data structure management.
    * `scikit-learn` for TF-IDF vectorization and pairwise metrics.
    * `nltk` for stemming, tokenization, and POS tagging.
    * `spacy` for advanced dependency parsing and linguistic analysis.
    * `sentence-transformers` for semantic embeddings and similarity.
    * `elasticsearch` for scalable indexing and vector search.
