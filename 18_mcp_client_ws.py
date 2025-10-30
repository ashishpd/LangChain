import asyncio
import json
from typing import Any, Dict

import websockets

WS_URL = "ws://localhost:8765/ws"


async def send(ws, msg: Dict[str, Any]) -> None:
    await ws.send(json.dumps(msg))


async def recv(ws) -> Dict[str, Any]:
    raw = await ws.recv()
    return json.loads(raw)


async def main() -> None:
    async with websockets.connect(WS_URL) as ws:
        # 1) initialize
        await send(
            ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        print("initialize ->", await recv(ws))

        # 2) tools/list
        await send(
            ws, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        )
        print("tools/list ->", await recv(ws))

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
