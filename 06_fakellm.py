from langchain_community.llms import FakeListLLM

# This LLM will always return the first unused item from responses.
llm = FakeListLLM(responses=["Hello from LangChain!"])

print(llm.invoke("Say hi"))