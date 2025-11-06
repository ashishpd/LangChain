"""
INTERVIEW STYLE Q&A:

Q: Why do you need output parsers in LangChain?
A: LLMs return message objects with metadata, but often you just want the text content.
   Output parsers extract and format the response, converting message objects to strings,
   structured data, or other formats you need.

Q: What does StrOutputParser do?
A: StrOutputParser extracts the text content from LLM response objects. It converts
   message objects (which have .content, .role, etc.) into plain strings, which is
   what you usually want for simple text responses.

Q: How do you add an output parser to an LCEL chain?
A: Add it to the end of your chain using the pipe operator: prompt | llm | parser
   The parser receives the LLM's response and extracts/formats it before returning.

Q: What's the benefit of using parsers in chains?
A: Parsers provide a clean separation - the LLM generates responses, the parser formats them.
   This makes your code more modular and allows you to swap parsers (e.g., JSON parser
   for structured output) without changing the rest of the chain.

SAMPLE CODE:
"""

from langchain_community.llms import FakeListLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Q: How do you create a prompt template?
# A: Use PromptTemplate.from_template() with {variable} placeholders
prompt = PromptTemplate.from_template("Answer briefly: {question}")

# Q: How do you set up a fake LLM for testing?
# A: Use FakeListLLM with predefined responses - useful for testing without API calls
llm = FakeListLLM(responses=["42"])

# Q: How do you create an output parser?
# A: Instantiate StrOutputParser() - it will extract text content from message objects
parser = StrOutputParser()

# Q: How do you create a complete chain with parsing?
# A: Use the pipe operator to connect: prompt → llm → parser
#    Data flows: input dict → prompt (fills template) → llm (generates response) → parser (extracts text) → final string
chain = prompt | llm | parser

# Q: How do you invoke the complete chain?
# A: Call invoke() with variable values - the chain handles everything and returns the parsed string
print(chain.invoke({"question": "What is the meaning of life?"}))
