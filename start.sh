#!/bin/bash

# MCP Memory Server Startup Script (with uv)

echo "Starting MCP Memory Server with uv..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv
fi

# Install/upgrade dependencies with uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

# Start Qdrant if not running
QDRANT_PORT=6333
if ! lsof -i:$QDRANT_PORT > /dev/null 2>&1; then
    echo "Starting Qdrant..."
    ./qdrant > logs/qdrant.log 2>&1 &
    sleep 3
else
    echo "Qdrant is already running on port $QDRANT_PORT"
fi

# Check Redis (optional)
REDIS_PORT=6379
if lsof -i:$REDIS_PORT > /dev/null 2>&1; then
    echo "Redis is running on port $REDIS_PORT"
else
    echo "Redis is not running (optional - cache will be disabled)"
fi

# Create necessary directories
mkdir -p data logs

# Start MCP server with uv
echo "Starting MCP Memory Server..."
echo "Server will be available for MCP clients to connect"
echo "Configuration: mcp.json"
echo "============================================"

# Run the MCP server with uv
uv run python mcp_memory_server.py