#!/bin/bash

echo "=================================================="
echo "  Elasticsearch + Semantic Search Quick Start"
echo "=================================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "Docker is running"
echo ""

# Start Elasticsearch
echo "Starting Elasticsearch container..."
docker compose up -d

echo ""
echo "Waiting for Elasticsearch to be ready (30 seconds)..."
sleep 5

# Check if Elasticsearch is up
for i in {1..6}; do
    if curl -s http://localhost:9200 > /dev/null 2>&1; then
        echo "Elasticsearch is ready!"
        break
    else
        echo "  Waiting... ($i/6)"
        sleep 5
    fi
done

# Verify connection
echo ""
if curl -s http://localhost:9200 > /dev/null 2>&1; then
    echo "Successfully connected to Elasticsearch"
    echo ""
    echo "=================================================="
    echo "  Elasticsearch is ready!"
    echo "=================================================="
    echo ""
    echo "Now run:"
    echo "  python src/main_elasticsearch.py"
    echo ""
    echo "To stop Elasticsearch later:"
    echo "  docker compose down"
    echo ""
else
    echo "Failed to connect to Elasticsearch"
    echo ""
    echo "Try:"
    echo "1. Check Docker logs: docker compose logs"
    echo "2. Restart: docker compose restart"
    echo "3. Check port 9200 is not in use"
fi
