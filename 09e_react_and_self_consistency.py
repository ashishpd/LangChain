"""
INTERVIEW STYLE Q&A:

Q: What is the ReAct (Reasoning + Acting) pattern?
A: ReAct is an agent pattern where the LLM: (1) Reasons about what to do (Thought),
   (2) Takes an action like calling a tool (Action), (3) Observes the result (Observation),
   (4) Continues reasoning or provides a final answer. It combines thinking with tool use.

Q: What is self-consistency and why is it useful?
A: Self-consistency means generating multiple answers to the same question and choosing
   the most common one (majority vote). This improves reliability because if the model
   gives the same answer multiple times, it's more likely to be correct.

Q: How do you implement a simple ReAct loop?
A: (1) Prompt the model to think and optionally call a tool, (2) Parse the response to
   detect tool calls, (3) Execute the tool, (4) Feed the observation back to get the
   final answer. This is a simplified version - LangGraph provides more robust implementations.

Q: When would you use self-consistency?
A: Use it for: important decisions, when accuracy matters more than speed, or when you
   can afford multiple API calls. It's especially useful for reasoning tasks where a single
   answer might be wrong but the majority is likely correct.

Note: This is a minimal, commented demonstration; for production, prefer LangGraph.

SAMPLE CODE:
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
