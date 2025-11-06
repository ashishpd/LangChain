"""
INTERVIEW STYLE Q&A:

Q: What is prompt routing?
A: Prompt routing uses an LLM to classify user questions and route them to different
   processing pipelines. For example, HR questions go to HR pipeline, policy questions
   go to policy pipeline, and hybrid questions use both.

Q: Why use routing instead of processing everything the same way?
A: Different question types need different data sources, tools, or processing logic.
   Routing ensures each question is handled by the most appropriate pipeline, improving
   accuracy and efficiency.

Q: How do you implement routing with prompts?
A: Create a classification prompt that asks the LLM to categorize the question into
   predefined routes. The prompt includes clear criteria and examples for each route.
   This is a lightweight approach - for production, consider LangGraph or tool-calling agents.

Q: What makes a good routing prompt?
A: Clear criteria for each route, examples showing correct classifications, and handling
   edge cases (like "unknown" for unrelated questions). The prompt should be unambiguous
   so the model consistently routes questions correctly.

SAMPLE CODE:
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
