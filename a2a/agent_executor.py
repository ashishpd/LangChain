"""
INTERVIEW STYLE Q&A:

Q: What is an AgentExecutor in A2A?
A: AgentExecutor is the interface that implements your agent's core logic. It receives
   requests via RequestContext, processes them, and sends responses via EventQueue.
   This is where you implement what your agent actually does.

Q: How do you implement an AgentExecutor?
A: Inherit from AgentExecutor and implement execute() method. Extract input from
   context.message, process it (call LLM, tools, etc.), then enqueue response
   messages using event_queue.enqueue_event().

Q: What is RequestContext?
A: RequestContext contains the incoming message and request metadata. It provides
   access to user input, conversation history, and other request-specific information
   your agent needs to process the request.

Q: What is EventQueue?
A: EventQueue is how agents send responses back to clients. You enqueue events
   (like text messages) which are then streamed to the client. This enables
   streaming responses and real-time communication.

SAMPLE CODE:
"""

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
