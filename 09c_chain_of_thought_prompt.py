"""
INTERVIEW STYLE Q&A:

Q: What is Chain of Thought (CoT) prompting?
A: Chain of Thought is a technique that encourages the LLM to show its reasoning process
   step-by-step before giving the final answer. Instead of jumping to a conclusion, the
   model breaks down the problem and explains each step of its thinking.

Q: Why is Chain of Thought effective?
A: Research shows that asking models to "think step by step" improves accuracy on complex
   problems, especially math, logic, and multi-step reasoning tasks. It helps the model
   avoid common errors by forcing it to work through the problem systematically.

Q: How do you implement Chain of Thought in LangChain?
A: Create a prompt template that explicitly asks the model to show reasoning, then use
   it in an LCEL chain. The prompt should include instructions like "think step by step"
   or "show your work".

Q: What's the difference between Chain of Thought and Tree of Thought?
A: Chain of Thought follows one reasoning path step-by-step. Tree of Thought explores
   multiple solution paths in parallel, evaluates them, and selects the best. CoT is
   simpler and faster; ToT is more thorough but uses more API calls.

SAMPLE CODE:
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

# Q: How do you set up the LLM for Chain of Thought reasoning?
# A: Create your LLM instance - it will be used to generate step-by-step reasoning
# Set up the LLM with Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: How do you design a Chain of Thought prompt?
# A: Include explicit instructions to "think step by step" and "show reasoning"
#    The prompt should guide the model to break down the problem before answering
# Chain of Thought Prompt Template
template = """
You are a helpful assistant.

Let's work through the problem step by step.

Problem: {input}

Think step by step, and show your reasoning before giving the answer.
A:
"""

# Q: How do you create the prompt template?
# A: Use PromptTemplate with your template string and specify input variables
prompt = PromptTemplate(
    input_variables=["input"],
    template=template,
)

# Q: Why use an output parser?
# A: StrOutputParser extracts clean text from the LLM's message response
parser = StrOutputParser()


# Q: How do you execute Chain of Thought reasoning?
# A: Create an LCEL chain (prompt | llm | parser) and invoke it with the problem
def run_chain_of_thought(user_input: str) -> str:
    # Q: How do you compose the chain?
    # A: Use the pipe operator to connect: prompt (fills template) → llm (generates reasoning) → parser (extracts text)
    # Compose the prompt with the input
    chain = prompt | llm | parser
    return chain.invoke({"input": user_input})


if __name__ == "__main__":
    result = run_chain_of_thought(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    )
    print(result)
