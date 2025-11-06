"""
INTERVIEW STYLE Q&A:

Q: What is LangGraph and how does it differ from regular LangChain chains?
A: LangGraph is a library for building stateful, multi-actor applications with LLMs.
   Unlike simple chains, LangGraph allows you to define graphs with nodes, edges, and
   state that persists between steps. It's ideal for agents, workflows, and complex
   multi-step processes.

Q: What is a StateGraph in LangGraph?
A: StateGraph is a graph where nodes share a common state object. Each node receives
   the state, processes it, and returns an updated state. This allows information to
   flow between nodes and enables complex workflows.

Q: How do you define state in LangGraph?
A: Use TypedDict to define your state structure. This ensures type safety and makes
   it clear what data flows between nodes. The state is passed to each node and can
   be modified.

Q: What are the basic components of a LangGraph?
A: (1) State definition (TypedDict), (2) Node functions that process state,
   (3) Graph construction (add nodes, set entry/finish points), (4) Compilation
   to create an executable app, (5) Invocation with initial state.

SAMPLE CODE:
"""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# Q: How do you define the state structure?
# A: Create a TypedDict class - this defines what data flows between nodes
#    This defines the object that will be passed between each node
class AgentState(TypedDict):
    messages: List[BaseMessage]


# Q: How do you create a node function?
# A: Define a function that takes state, processes it, and returns updated state
#    This defines the node that will be used to process the messages
def agent_node(state: AgentState) -> AgentState:
    # Q: How do you access state in a node?
    # A: Access state like a dictionary - state["messages"] gets the messages list
    print("Agent received messages:", state["messages"])
    # Q: How do you return updated state?
    # A: Return a dictionary with the same keys as your TypedDict
    return {"messages": state["messages"]}


# Q: How do you build a LangGraph?
# A: Create a StateGraph with your state type, add nodes, and set entry/finish points
# This defines the graph that will be used to connect the nodes
graph = StateGraph(AgentState)
# Q: How do you add a node to the graph?
# A: Use add_node() with a name and the node function
graph.add_node("agent", agent_node)
# Q: How do you set where the graph starts?
# A: Use set_entry_point() to specify which node runs first
graph.set_entry_point("agent")
# Q: How do you set where the graph ends?
# A: Use set_finish_point() to specify which node is the final one
graph.set_finish_point("agent")

# Q: How do you make the graph executable?
# A: Call compile() to create an app that can be invoked
# This defines the app that will be used to run the graph
app = graph.compile()

# Q: How do you run the graph?
# A: Call invoke() with initial state - the graph processes nodes in order
# This runs the app
result = app.invoke({"messages": [HumanMessage(content="Hello, how are you?")]})
print(result["messages"][-1].content)
