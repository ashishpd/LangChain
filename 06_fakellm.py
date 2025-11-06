"""
INTERVIEW STYLE Q&A:

Q: How do you test LLM applications without making actual API calls?
A: Use a FakeListLLM - a mock LLM that returns predefined responses from a list.
   This is useful for testing, development, and avoiding API costs during development.

Q: When would you use a fake LLM instead of a real one?
A: (1) Unit testing - test your code logic without API dependencies, (2) Development -
   iterate quickly without waiting for API calls or spending credits, (3) CI/CD pipelines -
   run tests without API keys or network access, (4) Demonstrations - show code structure
   without requiring API setup.

Q: How does FakeListLLM work?
A: It maintains a list of responses and returns them in order. Each time you invoke it,
   it returns the next response from the list. Once all responses are used, it cycles back
   or raises an error depending on configuration.

Q: What's the benefit of using LangChain's fake LLM vs mocking the API directly?
A: FakeListLLM implements the same interface as real LangChain LLMs, so your code doesn't
   need to change when switching between fake and real LLMs. This makes testing seamless.

SAMPLE CODE:
"""

from langchain_community.llms import FakeListLLM

# Q: How do you create a fake LLM for testing?
# A: Instantiate FakeListLLM with a list of responses - it will return these responses
#    in order when invoked, allowing you to test your code without making real API calls
#    This LLM will always return the first unused item from responses.
llm = FakeListLLM(responses=["Hello from LangChain!"])

# Q: How do you use the fake LLM?
# A: Call invoke() just like you would with a real LLM - the interface is identical
#    This allows you to test your application logic without depending on external APIs
print(llm.invoke("Say hi"))
