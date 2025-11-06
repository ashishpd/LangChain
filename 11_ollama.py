"""
INTERVIEW STYLE Q&A:

Q: What is Ollama and why would you use it?
A: Ollama is a tool for running large language models locally on your machine. It allows
   you to use LLMs without API calls, costs, or internet connectivity. Great for development,
   privacy-sensitive applications, or when you want full control.

Q: How do you use Ollama models with LangChain?
A: Use ChatOllama from langchain_ollama, which provides the same interface as other
   LangChain chat models. Just specify the model name (e.g., "gemma3:270m") and use
   it like any other LangChain LLM.

Q: What models are available in Ollama?
A: Ollama supports many models including Llama, Mistral, Gemma, and others. You install
   them using `ollama pull <model-name>`. The model name format is usually "model:version".

Q: What are the trade-offs of using local models vs cloud APIs?
A: Local models: No API costs, complete privacy, works offline, but requires local compute
   resources and may be slower/less capable than cloud models. Cloud APIs: More powerful,
   faster, but cost money and require internet.

SAMPLE CODE:
"""

from langchain_ollama import ChatOllama

# Q: How do you create a ChatOllama instance?
# A: Instantiate ChatOllama with the model name - this connects to your local Ollama instance
#    Make sure you've installed the model first: ollama pull gemma3:270m
llm = ChatOllama(model="gemma3:270m")

# Q: How do you use the local Ollama model?
# A: Same interface as other LangChain models - call invoke() with your prompt
#    The model runs locally, so no API calls or costs
print(llm.invoke("Say hi"))
