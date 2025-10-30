import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

parser = StrOutputParser()

template = """
Step1 :
 
I have a problem related to {input}. Could you brainstorm three distinct solutions? Please consider a variety of factors such as {perfect_factors}
A:
"""

prompt1 = PromptTemplate(
    input_variables=["input", "perfect_factors"], template=template
)

template = """
Step 2:

For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors

{solutions}

A:"""

prompt2 = PromptTemplate(input_variables=["solutions"], template=template)

template = """
Step 3:

For each solution, deepen the thought process. Generate potential scenarios, strategies for implementation, any necessary partnerships or resources, and how potential obstacles might be overcome. Also, consider any potential unexpected outcomes and how they might be handled.

{review}

A:"""

prompt3 = PromptTemplate(input_variables=["review"], template=template)

template = """
Step 4:

Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution
{deepen_thought_process}

A:"""

prompt4 = PromptTemplate(input_variables=["deepen_thought_process"], template=template)


def run_tree_of_thought(user_input: str, factors: str) -> str:
    solutions = (prompt1 | llm | parser).invoke(
        {
            "input": user_input,
            "perfect_factors": factors,
        }
    )

    review = (prompt2 | llm | parser).invoke(
        {
            "solutions": solutions,
        }
    )

    deepen = (prompt3 | llm | parser).invoke(
        {
            "review": review,
        }
    )

    ranked = (prompt4 | llm | parser).invoke(
        {
            "deepen_thought_process": deepen,
        }
    )

    return ranked


if __name__ == "__main__":
    result = run_tree_of_thought(
        user_input="choosing a project management tool for a small startup",
        factors="cost, onboarding time, integrations, AI features, mobile support",
    )
    print(result)
