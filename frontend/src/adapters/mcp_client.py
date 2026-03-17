import asyncio
from typing import AsyncGenerator
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool, CallToolResult

from config import settings
import logging

logger = logging.getLogger(__name__)

class MCPSubjectMatterClient:
    """Client for the remote mcp-subject-matter SSE server."""
    
    def __init__(self):
        self._url = settings.mcp_subject_matter_url
        self._tools: list[Tool] | None = None

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        """Context manager to establish an SSE connection and yield a session."""
        async with sse_client(self._url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session

    async def get_tools(self) -> list[dict]:
        """Fetch available tools from the MCP server and format them for OpenAI."""
        if self._tools is None:
            try:
                async with self._session() as session:
                    response = await session.list_tools()
                    self._tools = response.tools
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server at {self._url}: {e}")
                return []

        # Convert MCP Tool schema to OpenAI tool schema
        openai_tools = []
        for tool in self._tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            })
        return openai_tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool on the remote MCP server."""
        try:
            async with self._session() as session:
                result: CallToolResult = await session.call_tool(name, arguments)
                if result.isError:
                    return f"Error: {result.content}"
                
                # Combine TEXT responses
                texts = []
                for content in result.content:
                    if content.type == "text":
                        texts.append(content.text)
                text = "\n".join(texts).strip()
                # Treat empty or bare empty-list responses as no results
                if not text or text == "[]":
                    return f"No results found for {name}({arguments}). Answer from your own knowledge."
                return text
        except Exception as e:
            return f"Error executing tool {name}: {e}"
