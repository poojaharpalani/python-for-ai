import asyncio
import json
import os
import sys
from pathlib import Path
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv, find_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Load environment variables automatically
load_dotenv(find_dotenv(), override=True)


class MCPOpenAIClient:
    """Client for interacting with OpenAI-compatible models (Gemini) using MCP tools."""

    def __init__(self, model: str = "gemini-3.8-flash"):
        """Initialize the MCP client with Google's OpenAI-compatible endpoint."""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        gemini_host = "generativelanguage" + ".googleapis.com"
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url=f"https://{gemini_host}/v1beta/openai/",
            timeout=60.0,
            max_retries=2
        )

        self.model = model
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """Connect to an MCP server."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(SCRIPT_DIR / server_script_path)],
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        tools_result = await self.session.list_tools()
        print("\nConnected to server with tools:")
        for tool in tools_result.tools:
            print(f"  - {tool.name}: {tool.description}")

    async def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server in OpenAI format."""
        tools_result = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available MCP tools."""
        print("[LOG 1] Fetching MCP tools...", flush=True)
        tools = await self.get_mcp_tools()
        print(f"[LOG 2] Loaded {len(tools)} tool(s). Sending query to {self.model}...", flush=True)

        response = await asyncio.wait_for(
            self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                tools=tools,
                tool_choice="auto",
            ),
            timeout=60.0,
        )
        print("[LOG 3] Received initial response from Gemini.", flush=True)

        assistant_message = response.choices[0].message

        messages = [
            {"role": "user", "content": query},
            assistant_message,
        ]

        if assistant_message.tool_calls:
            print(f"[LOG 4] Gemini requested {len(assistant_message.tool_calls)} tool call(s).", flush=True)
            for tool_call in assistant_message.tool_calls:
                print(f"[LOG 5] Calling MCP tool '{tool_call.function.name}' with args: {tool_call.function.arguments}", flush=True)
                result = await asyncio.wait_for(
                    self.session.call_tool(
                        tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    ),
                    timeout=10.0,
                )
                print(f"[LOG 6] Tool returned ({len(result.content[0].text)} chars).", flush=True)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content[0].text,
                    }
                )

            print("[LOG 7] Sending tool output back to Gemini for final answer...", flush=True)
            final_response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                ),
                timeout=60.0,
            )
            print("[LOG 8] Final answer received!", flush=True)

            return final_response.choices[0].message.content

        print("[LOG 4b] No tool calls requested; returning direct response.", flush=True)
        return assistant_message.content


    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.exit_stack.aclose()
        finally:
            await self.openai_client.close()


async def main():
    """Main entry point for the client."""
    client = MCPOpenAIClient()
    try:
        await client.connect_to_server("server.py")

        query = "What is MCP?"
        print(f"\nQuery: {query}")

        response = await client.process_query(query)
        print(f"\nResponse: {response}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
