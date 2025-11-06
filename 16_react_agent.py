"""
INTERVIEW STYLE Q&A:

Q: What is a ReAct agent and how does it work?
A: ReAct (Reasoning + Acting) is an agent pattern where the LLM reasons about what to do,
   takes actions (like searching the web), observes results, and continues until it can
   answer the question. It combines reasoning with tool use.

Q: What are tools in the context of LangChain agents?
A: Tools are functions the agent can call to perform actions (search web, run code, query
   database, etc.). The agent decides when and how to use tools based on the user's question.

Q: How does LangGraph's create_react_agent work?
A: It creates a pre-built agent that follows the ReAct pattern: (1) Think about the question,
   (2) Decide if a tool is needed, (3) Call the tool, (4) Observe results, (5) Repeat or answer.
   It handles the orchestration automatically.

Q: What is memory/checkpointing in agents?
A: Checkpointing stores conversation state, allowing the agent to remember previous interactions
   in the same thread. This enables multi-turn conversations where the agent maintains context.

SAMPLE CODE:
"""

import os

from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Q: How do you set up the LLM for an agent?
# A: Create your chat model - it will be used for reasoning and decision-making
# 1. Instantiate the model
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: How do you provide tools to the agent?
# A: Create a list of tools - each tool is a callable function the agent can use
#    TavilySearch is a web search tool that the agent can call when it needs information
# 2. Instantiate the search tool
tools = [TavilySearch(max_results=2)]

# Q: How do you enable conversation memory?
# A: Use MemorySaver as a checkpointer - it stores conversation state in memory
#    This allows the agent to remember previous messages in the same conversation thread
# 3. Configure memory for conversational history
memory = MemorySaver()

# Q: How do you create a ReAct agent?
# A: Use create_react_agent() with your LLM, tools, and memory/checkpointer
#    This creates a ready-to-use agent that can reason and use tools
# 4. Create the agent with the model, tools, and memory
app = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
)

# Q: How do you invoke the agent with a query?
# A: Call invoke() with messages and a config containing a thread_id
#    The thread_id groups messages into the same conversation, enabling memory
# Invoke the agent with a query
config = {"configurable": {"thread_id": "conversation_1"}}
response = app.invoke(
    {
        "messages": [
            (
                "user",
                "What is the capital of France? And what is the weather like there?",
            )
        ]
    },
    config=config,
)

# Q: How do you access the agent's response?
# A: The response contains a messages list - the last message is the agent's final answer
# Print the final output
print(response["messages"][-1].content)

# Q: How do you continue the conversation in the same thread?
# A: Use the same config (thread_id) - the agent will remember previous messages
#    This enables follow-up questions that reference earlier context
# Invoke with a follow-up question in the same thread
response = app.invoke(
    {"messages": [("user", "what languages are spoken there")]},
    config=config,
)

print(response["messages"][-1].content)

# Q: How does the agent handle context from previous messages?
# A: Because we use the same thread_id, the agent has access to all previous messages
#    It can reference earlier information (like "there" referring to France's capital)
response = app.invoke(
    {"messages": [("user", "3 tourist places")]},
    config=config,
)

print(response["messages"][-1].content)
