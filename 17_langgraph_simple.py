from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from typing_extensions import TypedDict


# This defines the object that will be passed between each node
class AgentState(TypedDict):
    messages: List[BaseMessage]


# This defines the node that will be used to process the messages
def agent_node(state: AgentState) -> AgentState:
    print("Agent received messages:", state["messages"])
    return {"messages": state["messages"]}


# This defines the graph that will be used to connect the nodes
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.set_finish_point("agent")

# This defines the app that will be used to run the graph
app = graph.compile()

# This runs the app
result = app.invoke({"messages": [HumanMessage(content="Hello, how are you?")]})
print(result["messages"][-1].content)
