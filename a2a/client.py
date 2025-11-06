"""
INTERVIEW STYLE Q&A:

Q: How do you create an A2A client to interact with agents?
A: (1) Fetch the agent's AgentCard to discover capabilities, (2) Create a ClientFactory
   with configuration, (3) Use the factory to create a client from the AgentCard,
   (4) Send messages and receive responses asynchronously.

Q: What is an AgentCardResolver?
A: AgentCardResolver fetches AgentCard metadata from an agent server. The card contains
   information about the agent's capabilities, skills, and how to communicate with it.
   This enables dynamic discovery of agent capabilities.

Q: How do you send messages to an A2A agent?
A: Build a message dictionary with role, parts (content), and messageId. Use
   client.send_message() which returns an async iterator of events. Process events
   as they arrive for streaming responses.

Q: What's the benefit of async message handling?
A: Async enables: (1) Non-blocking I/O (handle multiple agents concurrently),
   (2) Streaming responses (process events as they arrive), (3) Better resource
   utilization, (4) Scalability for multi-agent systems.

SAMPLE CODE:
"""

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
