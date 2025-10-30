# agent_executor.py

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.utils import new_agent_text_message

class HelloWorldAgent:
    """Simple Hello World agent logic."""
    async def invoke(self) -> str:
        return "Hello World"

class HelloWorldAgentExecutor(AgentExecutor):
    """AgentExecutor implementation for HelloWorldAgent."""
    def __init__(self):
        super().__init__()
        self.agent = HelloWorldAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Extract input if needed from context.message
        # but for this simple example we ignore user input
        result = await self.agent.invoke()
        # Enqueue the response message
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Cancellation not supported in this simple example
        raise Exception("cancel not supported")