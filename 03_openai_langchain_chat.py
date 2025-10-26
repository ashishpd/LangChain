"""
Example: Use LangChain to call OpenAI chat completion and print the response.

Requirements:
- pip install langchain openai
- Set OPENAI_API_KEY in your environment
"""

from langchain_openai import ChatOpenAI

# You can also use langchain.chat_models.ChatOpenAI in older versions

# Uses OPENAI_API_KEY from environment
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = "Explain AI in 2 lines."
response = llm.invoke(prompt)
print(response.content)
