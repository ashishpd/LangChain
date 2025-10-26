from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:270m")

print(llm.invoke("Say hi"))