"""
INTERVIEW STYLE Q&A:

Q: How do you interact with OpenAI's API directly without using LangChain?
A: You use the OpenAI Python SDK (openai library) to create a client, then call the
   chat.completions.create() method with your messages and model name.

Q: What are the key components needed to make an OpenAI API call?
A: You need: (1) An API key stored as an environment variable, (2) A client instance
   created with that key, (3) A model name (like "gpt-3.5-turbo"), and (4) A messages
   array containing the conversation.

Q: How do you structure messages for the chat completion API?
A: Messages are structured as dictionaries with "role" (user/assistant/system) and
   "content" (the actual message text). The API expects a list of these message objects.

Q: What's the difference between using OpenAI SDK directly vs LangChain?
A: The OpenAI SDK gives you direct control and is simpler for basic use cases, while
   LangChain provides abstraction layers, chains, and additional features like memory
   management and tool integration.

SAMPLE CODE:
"""

import os

from openai import OpenAI

# Q: How do you securely access API keys in Python?
# A: Use environment variables to avoid hardcoding sensitive keys in your code
#    This prevents accidentally committing secrets to version control
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

# Q: How do you initialize the OpenAI client?
# A: Create a client instance by passing your API key to the OpenAI() constructor
#    This client will be used for all subsequent API calls
client = OpenAI(api_key=api_key)

# Q: How do you make a chat completion request?
# A: Call client.chat.completions.create() with:
#    - model: The model name (e.g., "gpt-3.5-turbo" or "gpt-4")
#    - messages: A list of message dictionaries with role and content
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain AI in 2 lines"}],
)

# Q: How do you extract the response text from the API response?
# A: Access response.choices[0].message.content - the API returns a structured object
#    where the actual text is nested in choices[0].message.content
#    Use .strip() to remove any leading/trailing whitespace
content = response.choices[0].message.content.strip()
print(content)
