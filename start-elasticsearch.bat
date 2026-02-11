@echo off
echo ==================================================
echo   Elasticsearch + Semantic Search Quick Start
echo ==================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running!
    echo Please start Docker Desktop and try again.
    exit /b 1
)

echo [OK] Docker is running
echo.

REM Start Elasticsearch
echo Starting Elasticsearch container...
docker compose up -d

echo.
echo Waiting for Elasticsearch to be ready (30 seconds)...
timeout /t 10 /nobreak >nul

REM Check if Elasticsearch is up
echo Checking Elasticsearch connection...
curl -s http://localhost:9200 >nul 2>&1
if errorlevel 1 (
    echo Waiting more...
    timeout /t 20 /nobreak >nul
    curl -s http://localhost:9200 >nul 2>&1
)

echo.
curl -s http://localhost:9200 >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to connect to Elasticsearch
    echo.
    echo Try:
    echo 1. Check Docker logs: docker compose logs
    echo 2. Restart: docker compose restart
    echo 3. Check port 9200 is not in use
) else (
    echo [OK] Successfully connected to Elasticsearch
    echo.
    echo ==================================================
    echo   Elasticsearch is ready!
    echo ==================================================
    echo.
    echo Now run:
    echo   python src/main_elasticsearch.py
    echo.
    echo To stop Elasticsearch later:
    echo   docker compose down
    echo.
)

pause
