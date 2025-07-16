#!/bin/bash

# Stop MCP Memory Server and related services

echo "Stopping MCP Memory Server..."

# Stop Qdrant
if pgrep -x "qdrant" > /dev/null; then
    echo "Stopping Qdrant..."
    pkill -x qdrant
    echo "Qdrant stopped"
else
    echo "Qdrant is not running"
fi

# Stop any Python MCP server processes
if pgrep -f "mcp_memory_server.py" > /dev/null; then
    echo "Stopping MCP server..."
    pkill -f "mcp_memory_server.py"
    echo "MCP server stopped"
else
    echo "MCP server is not running"
fi

echo "All services stopped"