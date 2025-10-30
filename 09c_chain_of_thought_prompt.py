import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

# Set up the LLM with Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Chain of Thought Prompt Template
template = """
You are a helpful assistant.

Let's work through the problem step by step.

Problem: {input}

Think step by step, and show your reasoning before giving the answer.
A:
"""

prompt = PromptTemplate(
    input_variables=["input"],
    template=template,
)

parser = StrOutputParser()


def run_chain_of_thought(user_input: str) -> str:
    # Compose the prompt with the input
    chain = prompt | llm | parser
    return chain.invoke({"input": user_input})


if __name__ == "__main__":
    result = run_chain_of_thought(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    )
    print(result)
