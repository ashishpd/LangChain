"""
INTERVIEW STYLE Q&A:

Q: How do you get structured JSON output from an LLM?
A: Ask the model explicitly to return JSON matching a schema, then parse and validate it.
   You can use Pydantic models to define the schema and validate the output automatically.

Q: Why validate LLM output with Pydantic?
A: LLMs sometimes return malformed JSON or data that doesn't match your schema. Pydantic
   validation catches these errors and ensures you get properly structured, typed data
   that matches your expectations.

Q: How do you handle validation failures?
A: Implement retry logic - if parsing or validation fails, ask the model to correct the
   output and try again. This is important because LLMs can make mistakes, but they can
   often fix them when given feedback.

Q: What's the benefit of using Pydantic Field constraints?
A: Field constraints (like pattern matching, min/max values) ensure data quality beyond
   just structure. For example, you can enforce that risk_level is only "low"/"medium"/"high"
   or that deadline_days is between 1-180.

SAMPLE CODE:
"""

import json
import os
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field, ValidationError


class ReleasePlan(BaseModel):
    title: str
    risk_level: str = Field(pattern=r"^(low|medium|high)$")
    owners: list[str]
    deadline_days: int = Field(ge=1, le=180)
    notes: Optional[str] = None


def ask_for_json(task: str, retries: int = 2) -> ReleasePlan:
    """Ask the LLM for JSON that matches ReleasePlan; retry if invalid."""
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
    )
    prompt = PromptTemplate.from_template(
        """
You are a planning assistant. Return ONLY a JSON object matching this schema:
{{
  "title": string,
  "risk_level": "low"|"medium"|"high",
  "owners": string[],
  "deadline_days": integer (1..180),
  "notes": string (optional)
}}
Task: {task}
        """.strip()
    )
    parser = StrOutputParser()

    last_err = None
    for _ in range(retries + 1):
        raw = (prompt | llm | parser).invoke({"task": task})
        try:
            data = json.loads(raw)
            return ReleasePlan.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            # Add a short correction hint and retry
            correction = PromptTemplate.from_template(
                "The last output failed to parse/validate: {err}\nPlease re-output valid JSON ONLY."
            )
            raw = (correction | llm | parser).invoke({"err": str(e)})
            try:
                data = json.loads(raw)
                return ReleasePlan.model_validate(data)
            except Exception:
                continue
    raise RuntimeError(f"Failed after retries: {last_err}")


if __name__ == "__main__":
    plan = ask_for_json("Create a small release plan to add SSO to our app")
    print(plan.model_dump_json(indent=2))
