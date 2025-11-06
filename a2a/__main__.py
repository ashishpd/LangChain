"""
INTERVIEW STYLE Q&A:

Q: What is A2A (Agent-to-Agent) and how does it work?
A: A2A is a protocol for agents to communicate with each other. It defines how agents
   advertise their capabilities (via AgentCard), how clients discover agents, and how
   messages are exchanged. It enables building multi-agent systems where agents
   collaborate.

Q: What is an AgentCard?
A: AgentCard is metadata that describes an agent's capabilities - what skills it has,
   what inputs/outputs it supports, and how to communicate with it. Clients use
   AgentCards to discover and interact with agents.

Q: How do you create an A2A server?
A: (1) Define AgentCard with skills and capabilities, (2) Create an AgentExecutor
   that implements your agent logic, (3) Wrap it in a RequestHandler, (4) Create
   A2AStarletteApplication with the card and handler, (5) Run with uvicorn.

Q: What's the benefit of using A2A?
A: A2A provides: (1) Standardized agent communication protocol, (2) Capability
   discovery (agents advertise what they can do), (3) Interoperability (agents from
   different systems can communicate), (4) Scalability (distributed agent networks).

SAMPLE CODE:
"""

# __main__.py
import uvicorn
from agent_executor import HelloWorldAgentExecutor  # your module

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


def make_agent_card(host: str = "localhost", port: int = 9999) -> AgentCard:
    # define the skill
    skill = AgentSkill(
        id="hello_world",
        name="Returns hello world",
        description="just returns hello world",
        tags=["hello world"],
        examples=["hi", "hello world"],
        inputModes=["text"],
        outputModes=["text"],
    )

    # define the agent card
    agent_card = AgentCard(
        name="Hello World Agent",
        description="Just a hello world agent",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )
    return agent_card


def main(host: str = "0.0.0.0", port: int = 9999):
    card = make_agent_card(host, port)

    # Setup the request handler that wraps your executor
    http_handler = DefaultRequestHandler(
        agent_executor=HelloWorldAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Instantiate the server app
    server_app = A2AStarletteApplication(
        agent_card=card,
        http_handler=http_handler,
    )

    # Run the ASGI server
    uvicorn.run(server_app.build(), host=host, port=port)


if __name__ == "__main__":
    main()
