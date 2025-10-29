import os

from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 1. Instantiate the model
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# 2. Instantiate the search tool
tools = [TavilySearch(max_results=2)]

# 3. Configure memory for conversational history
memory = MemorySaver()

# 4. Create the agent with the model, tools, and memory
app = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
)

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

# Print the final output
print(response["messages"][-1].content)

# Invoke with a follow-up question in the same thread
response = app.invoke(
    {"messages": [("user", "what languages are spoken there")]},
    config=config,
)

print(response["messages"][-1].content)

response = app.invoke(
    {"messages": [("user", "3 tourist places")]},
    config=config,
)

print(response["messages"][-1].content)
