#!/usr/bin/env python3
"""
MCP Memory Server - A Model Context Protocol server for memory management
Fully compliant with MCP specification
"""

import os
import json
import logging
from typing import Any
from datetime import datetime
import uuid
import asyncio

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from dotenv import load_dotenv
import redis
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import hashlib
import pickle

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("memory-server")

# Redis configuration
REDIS_ENABLED = False
redis_client = None

try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        decode_responses=False,
        socket_connect_timeout=5
    )
    redis_client.ping()
    REDIS_ENABLED = True
    logger.info("Redis cache connected successfully")
except Exception:
    logger.warning("Redis not available, running without cache")

# OpenAI client for embeddings
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Qdrant configuration
COLLECTION_NAME = "mcp_memories"
EMBEDDING_DIM = 3072

qdrant_client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

# Ensure collection exists
try:
    collections = qdrant_client.get_collections().collections
    if not any(col.name == COLLECTION_NAME for col in collections):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        logger.info(f"Created collection {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant: {e}")

# Initialize mem0
memory_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.2,
            "api_key": os.getenv('OPENAI_API_KEY')
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-large",
            "api_key": os.getenv('OPENAI_API_KEY')
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mcp_memories",
            "host": os.getenv('QDRANT_HOST', 'localhost'),
            "port": int(os.getenv('QDRANT_PORT', 6333)),
            "on_disk": True
        }
    },
    "history_db_path": "./data/mem0_history.db"
}

memory = Memory.from_config(memory_config)
logger.info("Mem0 service initialized successfully")

# Helper functions
def get_cache_key(prefix: str, content: str, user_id: str = None) -> str:
    """Generate cache key"""
    data = f"{content}:{user_id}" if user_id else content
    hash_key = hashlib.md5(data.encode()).hexdigest()
    return f"{prefix}:{hash_key}"

def get_from_cache(key: str) -> Any:
    """Get data from cache"""
    if not REDIS_ENABLED or not redis_client:
        return None
    try:
        cached = redis_client.get(key)
        if cached:
            return pickle.loads(cached)
    except Exception as e:
        logger.error(f"Cache read error: {e}")
    return None

def set_cache(key: str, value: Any, ttl: int = 300):
    """Set cache data"""
    if not REDIS_ENABLED or not redis_client:
        return
    try:
        redis_client.setex(key, ttl, pickle.dumps(value))
    except Exception as e:
        logger.error(f"Cache write error: {e}")

def generate_embedding(text: str) -> list[float]:
    """Generate embedding using OpenAI"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

# List available tools
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available tools"""
    return [
        types.Tool(
            name="add_memory",
            description="Add a new memory to the system",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store"},
                    "user_id": {"type": "string", "description": "User identifier", "default": "default_user"},
                    "metadata": {"type": "object", "description": "Additional metadata", "default": {}}
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="search_memories",
            description="Search memories using semantic search",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "user_id": {"type": "string", "description": "User identifier", "default": "default_user"},
                    "limit": {"type": "integer", "description": "Maximum results", "default": 10, "minimum": 1, "maximum": 100}
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="get_all_memories",
            description="Get all memories for a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "User identifier", "default": "default_user"}
                }
            }
        ),
        types.Tool(
            name="update_memory",
            description="Update an existing memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to update"},
                    "new_content": {"type": "string", "description": "New content for the memory"}
                },
                "required": ["memory_id", "new_content"]
            }
        ),
        types.Tool(
            name="delete_memory",
            description="Delete a memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory to delete"}
                },
                "required": ["memory_id"]
            }
        ),
        types.Tool(
            name="get_memory_history",
            description="Get history of changes for a memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of the memory"}
                },
                "required": ["memory_id"]
            }
        ),
        types.Tool(
            name="store_raw_memory",
            description="Store a memory directly without LLM processing",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Raw content to store"},
                    "user_id": {"type": "string", "description": "User identifier", "default": "default_user"},
                    "metadata": {"type": "object", "description": "Additional metadata", "default": {}}
                },
                "required": ["content"]
            }
        )
    ]

# Handle tool calls
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution"""
    
    if not arguments:
        arguments = {}
    
    try:
        if name == "add_memory":
            result = memory.add(
                messages=arguments.get("content"),
                user_id=arguments.get("user_id", "default_user"),
                metadata=arguments.get("metadata", {})
            )
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "memory_id": result.get("id"),
                    "message": "Memory added successfully"
                })
            )]
            
        elif name == "search_memories":
            query = arguments.get("query")
            user_id = arguments.get("user_id", "default_user")
            limit = arguments.get("limit", 10)
            
            # Check cache
            cache_key = get_cache_key("search", query, user_id)
            cached_results = get_from_cache(cache_key)
            
            if cached_results:
                logger.info(f"Cache hit for search: {query[:50]}...")
                results = cached_results
                from_cache = True
            else:
                results = memory.search(query=query, user_id=user_id, limit=limit)
                set_cache(cache_key, results, ttl=300)
                from_cache = False
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "results": results,
                    "cached": from_cache
                })
            )]
            
        elif name == "get_all_memories":
            user_id = arguments.get("user_id", "default_user")
            memories = memory.get_all(user_id=user_id)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "memories": memories,
                    "count": len(memories)
                })
            )]
            
        elif name == "update_memory":
            memory_id = arguments.get("memory_id")
            new_content = arguments.get("new_content")
            
            memory.update(memory_id=memory_id, data=new_content)
            
            # Clear search cache
            if REDIS_ENABLED and redis_client:
                try:
                    for key in redis_client.scan_iter("search:*"):
                        redis_client.delete(key)
                    logger.info("Cleared search cache after update")
                except Exception:
                    pass
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": "Memory updated successfully"
                })
            )]
            
        elif name == "delete_memory":
            memory_id = arguments.get("memory_id")
            memory.delete(memory_id=memory_id)
            
            # Clear search cache
            if REDIS_ENABLED and redis_client:
                try:
                    for key in redis_client.scan_iter("search:*"):
                        redis_client.delete(key)
                    logger.info("Cleared search cache after delete")
                except Exception:
                    pass
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "message": f"Memory {memory_id} deleted successfully"
                })
            )]
            
        elif name == "get_memory_history":
            memory_id = arguments.get("memory_id")
            history = memory.history(memory_id=memory_id)
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "history": history
                })
            )]
            
        elif name == "store_raw_memory":
            content = arguments.get("content")
            user_id = arguments.get("user_id", "default_user")
            metadata = arguments.get("metadata", {})
            
            # Check if already exists
            cache_key = get_cache_key("content", content, user_id)
            if get_from_cache(cache_key):
                return [types.TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "message": "Memory already exists",
                        "cached": True
                    })
                )]
            
            # Generate embedding
            embedding = generate_embedding(content)
            memory_id = str(uuid.uuid4())
            
            # Prepare metadata
            metadata.update({
                "user_id": user_id,
                "content": content,
                "created_at": datetime.now().isoformat()
            })
            
            # Store in Qdrant
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
            
            # Cache the content
            set_cache(cache_key, True, ttl=86400)  # 24 hours
            
            logger.info(f"Stored memory {memory_id} for user {user_id}")
            
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "memory_id": memory_id,
                    "message": "Memory stored successfully"
                })
            )]
            
        else:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Unknown tool: {name}"
                })
            )]
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e)
            })
        )]

# List available resources
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources"""
    return [
        types.Resource(
            uri="memory://stats",
            name="Memory Statistics",
            description="Get memory system statistics",
            mimeType="application/json",
        ),
        types.Resource(
            uri="memory://config",
            name="Memory Configuration",
            description="Get current memory configuration",
            mimeType="application/json",
        ),
    ]

# Handle resource reading
@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Handle resource reading"""
    if uri == "memory://stats":
        try:
            stats = {
                "total_memories": qdrant_client.count(collection_name=COLLECTION_NAME).count,
                "redis_enabled": REDIS_ENABLED,
                "embedding_model": "text-embedding-3-large",
                "vector_dimensions": EMBEDDING_DIM,
                "collection_name": COLLECTION_NAME
            }
            return json.dumps(stats, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    elif uri == "memory://config":
        config = {
            "llm_model": memory_config["llm"]["config"]["model"],
            "embedding_model": memory_config["embedder"]["config"]["model"],
            "vector_store": memory_config["vector_store"]["provider"],
            "cache_enabled": REDIS_ENABLED,
            "history_db": memory_config["history_db_path"]
        }
        return json.dumps(config, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")

# Main entry point
async def main():
    """Main entry point for the MCP server"""
    # Run the server using stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="memory-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())