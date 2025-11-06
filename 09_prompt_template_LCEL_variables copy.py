"""
INTERVIEW STYLE Q&A:

Q: What is LangChain Expression Language (LCEL) and why is it useful?
A: LCEL is a declarative way to compose chains using the pipe operator (|). It allows
   you to connect prompts, LLMs, and parsers in a readable, chainable way. Instead of
   writing verbose function calls, you can write: prompt | llm | parser.

Q: How do you create a prompt template with variables?
A: Use PromptTemplate.from_template() with curly braces {} to define variables. For
   example, "{topic}" becomes a variable that you can fill in when invoking the chain.

Q: What's the advantage of using LCEL's pipe operator?
A: It makes the data flow explicit and readable. You can see: prompt → llm → output
   in a single line. It also enables automatic streaming, batching, and async support.

Q: How do you invoke an LCEL chain with variables?
A: Call invoke() with a dictionary where keys match the variable names in your template.
   The chain will automatically fill in the variables and pass the result to the next step.

SAMPLE CODE:
"""

import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

# Q: How do you create a prompt template with a variable?
# A: Use PromptTemplate.from_template() with {variable_name} syntax
#    The {topic} will be replaced when you invoke the chain
prompt = PromptTemplate.from_template("Write a motto about {topic}.")

# Q: How do you set up the LLM for the chain?
# A: Create your LLM instance (AzureChatOpenAI in this case) - it will be used in the chain
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: How do you create an LCEL chain?
# A: Use the pipe operator (|) to connect components: prompt | llm
#    This creates a chain where: input → prompt (fills variables) → llm (generates response)
#    LCEL pipeline (LangChain Expression Language) - a declarative way to compose chains
chain = prompt | llm

# Q: How do you invoke the chain with variable values?
# A: Call invoke() with a dictionary - keys match template variables, values are what to fill in
#    The chain will: (1) Fill {topic} with "software craftsmanship", (2) Send to LLM, (3) Return response
print(chain.invoke({"topic": "software craftsmanship"}))
