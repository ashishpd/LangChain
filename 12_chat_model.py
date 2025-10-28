from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:270m")

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi! how are you?"),
]

response = llm.invoke(messages)
print(response.content)
