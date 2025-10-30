"""
09e_react_and_self_consistency.py

Demonstrates:
- ReAct-style loop (reason-act-observe) with a simple search/tool stub
- Self-consistency: sample multiple CoT answers and choose majority

Note: This is a minimal, commented demonstration; for production, prefer LangGraph.
"""

import os
from collections import Counter
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


# ---- Simple tool stub (replace with a real tool or retriever) ----
def fake_search(query: str) -> str:
    """Pretend to search the web and return a short snippet."""
    return f"[search result for: {query}]"


def react(question: str) -> str:
    """One-pass ReAct loop: model proposes an action, we execute, model answers."""
    system = (
        "You can think step-by-step (Thought), optionally call a tool (Action: search:<query>), "
        "then provide a concise Final Answer."
    )
    prompt = PromptTemplate.from_template(
        "{system}\n\nQuestion: {q}\nFormat:\nThought: ...\nAction: search:<query> (optional)\nObservation: <tool result> (if action used)\nFinal Answer: ..."
    )
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0.2
    )
    draft = (prompt | llm | StrOutputParser()).invoke({"system": system, "q": question})

    # Very naive parser: look for Action: search:<query>
    act_prefix = "Action: search:"
    if act_prefix in draft:
        query = draft.split(act_prefix, 1)[1].splitlines()[0].strip()
        obs = fake_search(query)
        # Feed observation to get final answer
        prompt2 = PromptTemplate.from_template(
            "{system}\n\nQuestion: {q}\n{draft}\nObservation: {obs}\nNow provide only Final Answer:"
        )
        return (prompt2 | llm | StrOutputParser()).invoke(
            {"system": system, "q": question, "draft": draft, "obs": obs}
        )

    # No action requested; extract Final Answer if present, else return draft
    if "Final Answer:" in draft:
        return draft.split("Final Answer:", 1)[1].strip()
    return draft


def self_consistency(question: str, samples: int = 5) -> str:
    """Sample multiple CoT answers and return the most common one (majority vote)."""
    cot_prompt = PromptTemplate.from_template(
        "Think step by step and then provide a short final answer.\nQuestion: {q}\nAnswer:"
    )
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0.7
    )
    parser = StrOutputParser()
    answers: List[str] = []
    for _ in range(samples):
        ans = (cot_prompt | llm | parser).invoke({"q": question})
        # Normalize lightly for voting
        answers.append(ans.strip().lower())
    [(best, _count)] = Counter(answers).most_common(1)
    return best


if __name__ == "__main__":
    q = "What is the fastest way to improve API reliability?"
    print("=== ReAct ===\n", react(q))
    print("\n=== Self-consistency (majority of 5) ===\n", self_consistency(q))
