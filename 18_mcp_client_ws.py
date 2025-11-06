"""
INTERVIEW STYLE Q&A:

Q: What is MCP (Model Context Protocol) and why use it?
A: MCP is a protocol for connecting AI applications to external tools and data sources
   via WebSockets. It enables agents to discover and call tools dynamically, making
   AI systems more extensible and capable.

Q: How does MCP work over WebSockets?
A: MCP uses JSON-RPC 2.0 messages over WebSocket connections. The client sends requests
   (initialize, tools/list, tools/call) and receives responses. This enables real-time
   bidirectional communication between AI agents and tool servers.

Q: What are the key MCP operations?
A: (1) initialize - establish connection and exchange capabilities, (2) tools/list -
   discover available tools, (3) tools/call - execute a tool with parameters. These
   follow JSON-RPC 2.0 format with method, params, and id fields.

Q: Why use WebSockets instead of HTTP for MCP?
A: WebSockets provide persistent, bidirectional connections ideal for real-time
   tool discovery and execution. They enable streaming responses and reduce connection
   overhead compared to HTTP request/response cycles.

SAMPLE CODE:
"""

import asyncio
import json
from typing import Any, Dict

import websockets

# Q: How do you specify the WebSocket server URL?
# A: Use ws:// for unencrypted or wss:// for encrypted WebSocket connections
WS_URL = "ws://localhost:8765/ws"


# Q: How do you send messages over WebSocket?
# A: Convert your data to JSON and send it using ws.send()
async def send(ws, msg: Dict[str, Any]) -> None:
    await ws.send(json.dumps(msg))


# Q: How do you receive messages over WebSocket?
# A: Use ws.recv() to get raw text, then parse JSON to get structured data
async def recv(ws) -> Dict[str, Any]:
    raw = await ws.recv()
    return json.loads(raw)


# Q: How do you establish a WebSocket connection?
# A: Use websockets.connect() as an async context manager
async def main() -> None:
    async with websockets.connect(WS_URL) as ws:
        # Q: How do you initialize an MCP connection?
        # A: Send an initialize request with JSON-RPC 2.0 format
        #    The server responds with capabilities and server info
        # 1) initialize
        await send(
            ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        print("initialize ->", await recv(ws))

        # Q: How do you discover available tools?
        # A: Send a tools/list request - the server responds with available tools
        #    and their schemas (parameters, descriptions, etc.)
        # 2) tools/list
        await send(
            ws, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        print("tools/list ->", await recv(ws))

        # Q: How do you call a tool?
        # A: Send a tools/call request with the tool name and arguments
        #    The server executes the tool and returns results
        # 3) tools/call (echo)
        await send(
            ws,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {"text": "hello mcp over websockets"},
                },
            },
        )
        print("tools/call ->", await recv(ws))


if __name__ == "__main__":
    asyncio.run(main())
