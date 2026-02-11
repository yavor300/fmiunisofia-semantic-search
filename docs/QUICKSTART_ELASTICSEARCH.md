# Quick Start with Elasticsearch

## Fastest Way to Get Started

### Windows

```bash
# 1. Start Elasticsearch
start-elasticsearch.bat

# 2. Run the search engine
python src/main_elasticsearch.py
```

### Linux/Mac

```bash
# 1. Start Elasticsearch
chmod +x start-elasticsearch.sh
./start-elasticsearch.sh

# 2. Run the search engine
python src/main_elasticsearch.py
```

## Manual Setup

### Step 1: Start Elasticsearch

```bash
docker compose up -d
```

Wait 30 seconds, then verify:
```bash
curl http://localhost:9200
```

### Step 2: Run the Indexer

```bash
python src/main_elasticsearch.py
```

This will:
1. Connect to Elasticsearch
2. Load product CSV
3. Index all products (~1000 products in dataset)
4. Start interactive search

### Step 3: Search

```
Search> running shoes
Search> Saucony running shoes under 60
Search> cheap safety vest
Search> solar lights outdoor
```

## What Gets Indexed

Each product becomes a searchable document:

- **Title**: Product name
- **Brand**: Manufacturer
- **Description**: Full product description  
- **Price**: Numeric price value
- **Category**: Product category
- **Rating**: Customer rating (1-5)
- **Reviews**: Number of reviews
- **Semantic Tags**: Auto-generated (budget, premium, etc.)
- **Availability**: Stock status

## Useful Commands

### Elasticsearch Management

```bash
# Start
docker compose up -d

# Stop
docker compose down

# Stop and remove data
docker compose down -v

# View logs
docker compose logs -f elasticsearch

# Restart
docker compose restart
```

### Index Management

```bash
# Check index exists
curl http://localhost:9200/_cat/indices

# Count documents
curl http://localhost:9200/amazon_products/_count

# Get mapping
curl http://localhost:9200/amazon_products/_mapping

# Delete index
curl -X DELETE http://localhost:9200/amazon_products
```

### Search via curl (for testing)

```bash
# Simple search
curl -X GET "http://localhost:9200/amazon_products/_search?q=running+shoes&pretty"

# Search with filters
curl -X POST "http://localhost:9200/amazon_products/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": {"match": {"title": "running shoes"}},
      "filter": {"range": {"price": {"lte": 60}}}
    }
  }
}'
```
