# MCP Memory Server

A Model Context Protocol (MCP) compliant memory management server for Claude Code.

## Features

- ✅ Full MCP specification compliance
- ✅ Intelligent memory management with mem0
- ✅ Vector storage with Qdrant
- ✅ Optional Redis caching
- ✅ OpenAI embeddings (text-embedding-3-large)
- ✅ Semantic search capabilities

## Quick Start

1. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY
   ```

2. **Start the server**:
   ```bash
   ./start.sh
   ```

3. **Stop the server**:
   ```bash
   ./stop.sh
   ```

## Project Structure

```
mem0-service/
├── mcp_memory_server.py  # Main MCP server
├── mcp.json             # MCP configuration
├── requirements.txt     # Python dependencies
├── start.sh            # Start script (uses uv)
├── stop.sh             # Stop script
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
├── README.md           # This file
├── data/               # Data storage (auto-created)
├── logs/               # Log files (auto-created)
└── .venv/              # Virtual environment (uv)
```

## MCP Tools

The server provides 7 memory management tools:

1. **add_memory** - Add new memories
2. **search_memories** - Semantic search with caching
3. **get_all_memories** - Retrieve all user memories
4. **update_memory** - Update existing memories
5. **delete_memory** - Delete memories
6. **get_memory_history** - View memory change history
7. **store_raw_memory** - Direct storage without LLM processing

## MCP Resources

- `memory://stats` - System statistics
- `memory://config` - Current configuration

## Configuration

Configure Claude Desktop to use this MCP server by adding to your MCP settings:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["mcp_memory_server.py"],
      "cwd": "/path/to/mem0-service"
    }
  }
}
```

## Dependencies

- Python 3.10+
- uv (for virtual environment management)
- Qdrant v1.14.1 (included)
- Redis (optional, for caching)

## License

MIT