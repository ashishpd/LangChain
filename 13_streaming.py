"""
INTERVIEW STYLE Q&A:

Q: What is streaming in the context of LLMs and why is it useful?
A: Streaming means receiving the response token-by-token as it's generated, rather than
   waiting for the complete response. This provides immediate feedback to users, making
   applications feel more responsive, especially for longer responses.

Q: How do you enable streaming with LangChain chat models?
A: Use the stream() method instead of invoke(). It returns an iterator that yields
   message chunks as they're generated. Each chunk contains part of the response.

Q: What's the difference between stream() and invoke()?
A: invoke() waits for the complete response and returns it all at once. stream() returns
   an iterator that yields chunks immediately as they're generated, allowing you to
   display partial responses in real-time.

Q: When would you use streaming vs regular invocation?
A: Use streaming for: user-facing applications (chat UIs), long responses, or when you
   want to show progress. Use regular invocation for: background processing, when you
   need the complete response before proceeding, or simpler use cases.

SAMPLE CODE:
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Q: How do you set up a chat model for streaming?
# A: Create your LLM instance normally - streaming is enabled by using stream() method
llm = ChatOllama(model="gemma3:270m")

# Q: How do you structure messages for streaming?
# A: Same as regular invocation - create a list of SystemMessage and HumanMessage objects
messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi! how are you?"),
]

# Q: How do you stream the response?
# A: Use stream() instead of invoke() - it returns an iterator that yields message chunks
#    Each iteration gives you a chunk with .content containing the next part of the response
#    The end="|" parameter prints tokens separated by | instead of newlines
for token in llm.stream(messages):
    print(token.content, end="|")
