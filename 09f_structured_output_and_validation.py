"""
09f_structured_output_and_validation.py

Demonstrates:
- Asking the model for JSON-structured output
- Validating with Pydantic
- Retrying on parse/validation failure
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
