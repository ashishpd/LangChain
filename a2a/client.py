# client.py
import asyncio
import uuid

import httpx

from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory


async def main():
    agent_url = "http://localhost:9999"  # adjust if needed

    async with httpx.AsyncClient() as httpx_client:
        # 1. Fetch agent card
        resolver = A2ACardResolver(httpx_client, base_url=agent_url)
        agent_card = await resolver.get_agent_card()

        # 2. Prepare client configuration
        config = ClientConfig(
            httpx_client=httpx_client
        )  # ensure HTTP client is part of config

        # 3. Create ClientFactory with config
        factory = ClientFactory(config=config, consumers=None)

        # 4. Create client using agent_card
        client = factory.create(agent_card)

        # 5. Build the message
        message = {
            "role": "user",
            "parts": [{"kind": "text", "text": "Hello agent, how are you?"}],
            "messageId": uuid.uuid4().hex,
        }

        # 6. Send message & iterate responses
        # send_message expects a Message or dict, not SendMessageRequest
        async for event in client.send_message(message):
            print("Received event:", event)

        # 7. Close client when done
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
