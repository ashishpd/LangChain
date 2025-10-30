# __main__.py
import uvicorn
from a2a.types import AgentSkill, AgentCapabilities, AgentCard
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from agent_executor import HelloWorldAgentExecutor  # your module

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