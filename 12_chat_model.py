"""
INTERVIEW STYLE Q&A:

Q: How do you structure messages for chat models in LangChain?
A: Use message classes from langchain_core.messages: SystemMessage for instructions/context,
   HumanMessage for user input, and AIMessage for assistant responses. Pass a list of
   these messages to the LLM's invoke() method.

Q: What's the difference between passing a string vs messages list to invoke()?
A: Passing a string is simpler for single-turn conversations. Passing a messages list
   gives you more control - you can include system messages, maintain conversation history,
   and structure multi-turn conversations properly.

Q: When would you use SystemMessage vs HumanMessage?
A: SystemMessage sets the model's role, instructions, or context (e.g., "You are a translator").
   HumanMessage contains the actual user input or question. System messages guide behavior,
   human messages are what the model responds to.

Q: How do chat models handle message lists?
A: Chat models process messages in order, understanding the roles. They use system messages
   for context, respond to human messages, and can maintain conversation history when
   you include previous exchanges.

SAMPLE CODE:
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Q: How do you initialize a chat model?
# A: Create an instance with your model name - works with ChatOllama, ChatOpenAI, etc.
llm = ChatOllama(model="gemma3:270m")

# Q: How do you structure messages for a chat model?
# A: Create a list of message objects:
#    - SystemMessage: Sets instructions/context for the model
#    - HumanMessage: Contains the user's input/question
#    Messages are processed in order, so system message comes first
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi! how are you?"),
]

# Q: How do you invoke the model with structured messages?
# A: Pass the messages list to invoke() - the model processes them in order
#    and generates a response based on the system instructions and user input
response = llm.invoke(messages)

# Q: How do you access the response content?
# A: The response is a message object with a .content attribute containing the text
print(response.content)
