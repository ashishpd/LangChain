"""
INTERVIEW STYLE Q&A:

Q: How do you make an LLM remember previous parts of a conversation?
A: Use LangChain's memory components, which store conversation history and automatically
   include it in subsequent prompts. This allows the LLM to maintain context across turns.

Q: What is ConversationBufferWindowMemory?
A: It's a memory type that keeps only the last N exchanges (conversation turns) in memory.
   The "window" (k parameter) limits how much history is retained, preventing the context
   from growing too large and managing token costs.

Q: What's the difference between different memory types?
A: - ConversationBufferMemory: Keeps all conversation history (can get expensive)
   - ConversationBufferWindowMemory: Keeps only last N exchanges (this example)
   - ConversationSummaryMemory: Summarizes old history to save tokens
   - Each has different trade-offs between context and cost

Q: How does ConversationChain work with memory?
A: ConversationChain automatically manages the conversation flow. It stores each exchange
   in memory and includes relevant history when making new predictions, so the LLM can
   reference previous messages.

SAMPLE CODE:
"""

import os

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI

# Q: How do you set up the LLM for conversations?
# A: Create your chat model instance - it will be used by the conversation chain
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: How do you create a memory component?
# A: Use ConversationBufferWindowMemory with k parameter - this keeps the last k exchanges
#    k=3 means it remembers the last 3 back-and-forth interactions
# Step 1: Initialize the memory with a window of 3 exchanges
memory = ConversationBufferWindowMemory(k=3)

# Q: How do you create a conversation chain with memory?
# A: Use ConversationChain with your LLM and memory - it automatically manages conversation flow
#    verbose=True shows the prompts being sent (useful for debugging)
# Step 2: Create a conversation chain with this memory
conversation = ConversationChain(
    llm=llm, memory=memory, verbose=False  # Set to True to see the prompt and responses
)

# Q: How do you have a multi-turn conversation?
# A: Call predict() multiple times with different inputs - the chain automatically includes
#    previous exchanges from memory in each new request
# Step 3: Try a conversation
print(conversation.predict(input="Hi, my name is Alice."))
print(conversation.predict(input="What is my name?"))  # Should remember "Alice"
print(conversation.predict(input="I like to play chess."))
print(conversation.predict(input="What was the last thing I said I liked to do?"))  # Should remember "chess"
print(conversation.predict(input="Where do I live?"))  # Not mentioned, won't know
print(conversation.predict(input="Do you remember my name?"))  # Should still remember "Alice" if within window
