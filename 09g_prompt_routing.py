"""
09g_prompt_routing.py

Demonstrates a lightweight routing prompt to choose which pipeline to use.
We simulate three routes: policy-only, hr-only, hybrid. In production, you might
replace this with a LangGraph node or tool-calling agent.
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

ROUTES = ["policy", "hr", "hybrid"]


def classify_route(question: str) -> str:
    """Return one of 'policy' | 'hr' | 'hybrid'."""
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
    )
    prompt = PromptTemplate.from_template(
        "Classify the user question into a single route: policy | hr | hybrid.\n"
        "Instructions:\n"
        "- 'policy': ONLY for company rules/policies (e.g., overtime policy, PTO accrual, leave eligibility, holiday policy, expense policy).\n"
        "- 'hr': ONLY for personal HR data (e.g., my manager, my salary, my PTO balance, my start date, my years of service).\n"
        "- 'hybrid': if BOTH policy and personal HR data are required.\n"
        "- If the question is unrelated to company policy or personal HR data (e.g., weather, sports, general trivia), return 'unknown'.\n\n"
        "Examples:\n"
        "- 'Who is my manager?' -> hr\n"
        "- 'How is overtime computed?' -> policy\n"
        "- 'Given my 2 years, what overtime am I eligible for?' -> hybrid\n"
        "Return only one word: policy | hr | hybrid | unknown.\n\n"
        "Question: {q}"
    )
    route = (prompt | llm | StrOutputParser()).invoke({"q": question}).strip().lower()
    return route if route in ROUTES else "I don't know."


if __name__ == "__main__":
    tests = [
        "What is my manager's name?",
        "How is overtime computed?",
        "Given my 2 years, what overtime am I eligible for?",
        "How's the weather in Tokyo?",
        "How's the capital of France?",
    ]
    for t in tests:
        print(t, "->", classify_route(t))
