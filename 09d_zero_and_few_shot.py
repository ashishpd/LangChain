"""
09d_zero_and_few_shot.py

Demonstrates: Zero-shot vs Few-shot prompting with LangChain LCEL.
Uses ChatOpenAI; adjust model/env as needed. Comments explain each step.
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


def zero_shot(task: str) -> str:
    """Simple instruction-only prompt (zero-shot)."""
    prompt = PromptTemplate.from_template(
        "You are concise. Perform the task and keep the answer under 6 sentences.\n\nTask: {task}"
    )
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"task": task})


def few_shot(task: str) -> str:
    """Add 3 exemplars to guide style/format (few-shot)."""
    examples = (
        "Example 1:\nInput: Summarize benefits of TDD\nOutput: TDD encourages small, testable units...\n\n"
        "Example 2:\nInput: Give 3 downsides of long functions\nOutput: They are harder to test...\n\n"
        "Example 3:\nInput: Write a short plan to learn SQL\nOutput: Start with SELECT...\n\n"
    )
    template = "Follow the style of the examples. Be concise.\n\n{examples}\nInput: {task}\nOutput:"
    prompt = PromptTemplate.from_template(template)
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"examples": examples, "task": task})


if __name__ == "__main__":
    t = "Provide a brief plan to improve code review quality at a startup"
    print("=== Zero-shot ===\n", zero_shot(t))
    print("\n=== Few-shot ===\n", few_shot(t))
