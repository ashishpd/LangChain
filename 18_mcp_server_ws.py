"""
INTERVIEW STYLE Q&A:

Q: How do you create an MCP server using WebSockets?
A: Use FastAPI's WebSocket support to create a WebSocket endpoint that handles
   JSON-RPC 2.0 messages. Implement handlers for initialize, tools/list, and
   tools/call methods following the MCP protocol.

Q: What is JSON-RPC 2.0 and why does MCP use it?
A: JSON-RPC 2.0 is a stateless, light-weight remote procedure call (RPC) protocol.
   It provides a standard format for requests (method, params, id) and responses
   (result or error). MCP uses it for tool discovery and execution.

Q: How do you handle different MCP methods on the server?
A: Parse incoming JSON-RPC messages, check the "method" field, and route to
   appropriate handlers. Each method (initialize, tools/list, tools/call) has
   specific request/response formats defined by the MCP protocol.

Q: What makes a valid MCP tool?
A: A tool needs: (1) A name, (2) A description, (3) An inputSchema defining
   parameters (JSON Schema format). The server advertises tools via tools/list
   and executes them via tools/call.

SAMPLE CODE:
"""

import json
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Sample MCP WebSocket Server")


def jsonrpc_result(id_value: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_value, "result": result}


def jsonrpc_error(
    id_value: Any, code: int, message: str, data: Any | None = None
) -> Dict[str, Any]:
    err: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": id_value,
        "error": {"code": code, "message": message},
    }
    if data is not None:
        err["error"]["data"] = data
    return err


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps(jsonrpc_error(None, -32700, "Parse error"))
                )
                continue

            method = msg.get("method")
            id_value = msg.get("id")
            params = msg.get("params") or {}

            if method == "initialize":
                # Minimal MCP initialize response
                capabilities = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": True,
                    },
                    "serverInfo": {
                        "name": "sample-mcp-ws",
                        "version": "0.1.0",
                    },
                }
                await websocket.send_text(
                    json.dumps(jsonrpc_result(id_value, capabilities))
                )

            elif method == "tools/list":
                # Advertise a single echo tool
                tools = {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back the provided text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                },
                                "required": ["text"],
                            },
                        }
                    ]
                }
                await websocket.send_text(json.dumps(jsonrpc_result(id_value, tools)))

            elif method == "tools/call":
                # params expected: { name: str, arguments: dict }
                name = (params or {}).get("name")
                arguments = (params or {}).get("arguments") or {}

                if name == "echo":
                    text = arguments.get("text", "")
                    result = {
                        "content": [
                            {
                                "type": "text",
                                "text": text,
                            }
                        ]
                    }
                    await websocket.send_text(
                        json.dumps(jsonrpc_result(id_value, result))
                    )
                else:
                    await websocket.send_text(
                        json.dumps(
                            jsonrpc_error(id_value, -32601, f"Unknown tool: {name}")
                        )
                    )

            else:
                await websocket.send_text(
                    json.dumps(jsonrpc_error(id_value, -32601, "Method not found"))
                )

    except WebSocketDisconnect:
        # Client disconnected
        return


def build() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("18_mcp_server_ws:app", host="0.0.0.0", port=8765, reload=False)
