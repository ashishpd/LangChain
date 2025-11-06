"""
INTERVIEW STYLE Q&A:

Q: How does LangChain simplify working with OpenAI's API?
A: LangChain provides a ChatOpenAI class that abstracts away the API details.
   Instead of manually creating clients and message structures, you just create
   a ChatOpenAI instance and call invoke() with a simple string prompt.

Q: What's the main advantage of using LangChain over the OpenAI SDK directly?
A: LangChain provides a consistent interface across different LLM providers (OpenAI,
   Anthropic, Azure, etc.), making it easy to switch providers. It also integrates
   with chains, memory, and other LangChain features.

Q: How do you use LangChain's ChatOpenAI class?
A: Import ChatOpenAI from langchain_openai, create an instance with your model name,
   then call invoke() with your prompt string. The API key is automatically read
   from the OPENAI_API_KEY environment variable.

Q: What's the difference between ChatOpenAI and the regular OpenAI client?
A: ChatOpenAI is LangChain's wrapper that provides a simpler interface and integrates
   with LangChain's ecosystem (chains, prompts, memory, etc.), while the OpenAI client
   is the direct SDK with more granular control.

Requirements:
- pip install langchain langchain-openai openai
- Set OPENAI_API_KEY in your environment

SAMPLE CODE:
"""

from langchain_openai import ChatOpenAI

# Q: How do you create a LangChain chat model instance?
# A: Instantiate ChatOpenAI with the model name - it automatically reads OPENAI_API_KEY
#    from environment variables, so you don't need to pass it explicitly
# Note: In older LangChain versions, you might use langchain.chat_models.ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Q: How do you invoke the model with LangChain?
# A: Simply call invoke() with a string prompt - no need to structure messages manually
#    LangChain handles the message formatting internally
prompt = "Explain AI in 2 lines."
response = llm.invoke(prompt)

# Q: How do you access the response content?
# A: The response object has a .content attribute that contains the text response
#    This is simpler than navigating the OpenAI API's nested response structure
print(response.content)
