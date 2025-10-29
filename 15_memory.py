import os

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Step 1: Initialize the memory with a window of 3 exchanges
memory = ConversationBufferWindowMemory(k=3)

# Step 2: Create a conversation chain with this memory
conversation = ConversationChain(
    llm=llm, memory=memory, verbose=False  # Set to True to see the prompt and responses
)

# Step 3: Try a conversation
print(conversation.predict(input="Hi, my name is Alice."))
print(conversation.predict(input="What is my name?"))
print(conversation.predict(input="I like to play chess."))
print(conversation.predict(input="What was the last thing I said I liked to do?"))
print(conversation.predict(input="Where do I live?"))
print(conversation.predict(input="Do you remember my name?"))
