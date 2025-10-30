import os
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import (AzureChatOpenAI, AzureOpenAIEmbeddings,
                              ChatOpenAI, OpenAIEmbeddings)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

POLICY_PDF = Path(__file__).parent / "21_policy_overtime.pdf"
PERSIST_BASE = str(Path(__file__).parent / ".chroma_hr_policy21")


def ensure_policy_pdf() -> None:
    if POLICY_PDF.exists():
        return
    text = (
        "Company Overtime & Leave Policy\n\n"
        "- Employees with 1 year of service receive 1.25x overtime pay.\n"
        "- Employees with exactly 2 years of service receive 1.5x overtime pay.\n"
        "- Employees with more than 2 years of service receive 1.7x overtime pay.\n"
        "- Paid Time Off (PTO) accrues per department policy and role.\n"
    )
    c = canvas.Canvas(str(POLICY_PDF), pagesize=letter)
    width, height = letter
    x, y = 72, height - 72
    for line in text.split("\n"):
        c.drawString(x, y, line)
        y -= 16
    c.showPage()
    c.save()


def make_embeddings():
    if os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"):
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
        )
    return OpenAIEmbeddings()


def make_llm():
    if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0,
        )
    return ChatOpenAI(temperature=0, model="gpt-4o-mini")


def build_or_load_index() -> Chroma:
    ensure_policy_pdf()
    embeddings = make_embeddings()
    deployment_suffix = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "openai")
    persist_dir = f"{PERSIST_BASE}_{deployment_suffix}"

    if Path(persist_dir).exists():
        try:
            return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        except Exception:
            pass

    loader = PyPDFLoader(str(POLICY_PDF))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = splitter.split_documents(pages)
    vs = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=persist_dir
    )
    return vs


app = FastAPI(title="HR Policy Server 21")
vectorstore = build_or_load_index()
llm = make_llm()
parser = StrOutputParser()


# Fake HR data (expandable)
USER_PROFILE: Dict[str, Dict[str, object]] = {
    "alice": {
        "years": 0.8,
        "dob": "1995-08-09",
        "title": "Analyst",
        "manager": "bob",
        "salary": 90000,
        "pto_balance": 32,
    },
    "bob": {
        "years": 1.0,
        "dob": "1990-01-05",
        "title": "Manager",
        "manager": "carol",
        "salary": 140000,
        "pto_balance": 18,
    },
    "carol": {
        "years": 2.0,
        "dob": "1988-02-10",
        "title": "Senior Manager",
        "manager": "dave",
        "salary": 170000,
        "pto_balance": 25,
    },
    "dave": {
        "years": 3.4,
        "dob": "1985-11-22",
        "title": "Director",
        "manager": None,
        "salary": 220000,
        "pto_balance": 12,
    },
}


def pick_multiplier(years: float) -> float:
    if years > 2:
        return 1.7
    if years == 2:
        return 1.5
    if years >= 1:
        return 1.25
    return 1.0


# ----- Intent Routing -----
def route_intent(q: str) -> str:
    ql = q.lower()
    hr_signals = [
        "date of birth",
        "dob",
        "manager",
        "title",
        "salary",
        "pto",
        "pto balance",
        "years of service",
        "start date",
        "email",
        "phone",
        "employee id",
    ]
    policy_signals = [
        "overtime",
        "leave policy",
        "travel policy",
        "expense policy",
        "holiday",
        "paid time off",
        "policy",
    ]

    if any(k in ql for k in hr_signals) and any(k in ql for k in policy_signals):
        return "hybrid_query"
    if any(k in ql for k in hr_signals):
        return "hr_query"
    if any(k in ql for k in policy_signals):
        return "policy_query"
    return "hybrid_query"


def redact_if_sensitive(key: str) -> bool:
    # Very simple privacy gate: deny DOB and salary unless explicitly allowed via env
    if key in {"dob", "salary"} and not os.getenv(
        "ALLOW_PII", ""
    ):  # empty means not allowed
        return True
    return False


async def fetch_hr_fields(user: str, fields: list[str]) -> Dict[str, object]:
    # In real life: call external HR APIs per field. Here we serve from USER_PROFILE.
    profile = USER_PROFILE.get(user.lower()) or {}
    data: Dict[str, object] = {}
    for f in fields:
        if redact_if_sensitive(f):
            continue
        if f == "years":
            data["years"] = float(profile.get("years", 0.0))
        elif f in profile:
            data[f] = profile[f]
    return data


FIELD_MAP = {
    "date of birth": ["dob"],
    "dob": ["dob"],
    "manager": ["manager"],
    "title": ["title"],
    "salary": ["salary"],
    "pto": ["pto_balance"],
    "pto balance": ["pto_balance"],
    "years of service": ["years"],
}


def fields_for_question(q: str) -> list[str]:
    ql = q.lower()
    fields: list[str] = []
    for k, vals in FIELD_MAP.items():
        if k in ql:
            fields.extend(vals)
    # Ensure multiplier dependency
    if "overtime" in ql and "years" not in fields:
        fields.append("years")
    # Dedupe
    seen = set()
    out: list[str] = []
    for f in fields:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


class AskRequest(BaseModel):
    user: str
    question: str


@app.get("/hr/profile/{user}")
async def hr_profile(user: str) -> Dict[str, object]:
    profile = USER_PROFILE.get(user.lower())
    if not profile:
        raise HTTPException(status_code=404, detail="user not found")
    safe = {k: v for k, v in profile.items() if not redact_if_sensitive(k)}
    return safe


@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, object]:
    intent = route_intent(req.question)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    policy_context = ""
    hr_facts: Dict[str, object] = {}

    if intent in ("policy_query", "hybrid_query"):
        docs = await retriever.ainvoke(req.question)
        policy_context = "\n\n".join(d.page_content for d in docs)

    if intent in ("hr_query", "hybrid_query"):
        needed_fields = fields_for_question(req.question)
        hr_facts = await fetch_hr_fields(
            req.user, needed_fields or ["years"]
        )  # default years for overtime

    # Select prompt by intent
    prompt_policy = PromptTemplate.from_template(
        "Answer using ONLY the policy context below. If missing, say you don't know.\n\n{context}\n\nQ: {question}"
    )
    prompt_hr = PromptTemplate.from_template(
        "Answer using ONLY these HR facts; do not infer or fabricate.\nFacts: {facts}\n\nQ: {question}"
    )
    prompt_hybrid = PromptTemplate.from_template(
        "Combine HR facts (for personal details) and policy context (for rules). If either is missing, say so explicitly.\n\nHR facts: {facts}\n\nPolicy: {context}\n\nQ: {question}"
    )

    if intent == "policy_query":
        chain = prompt_policy | llm | parser
        answer = chain.invoke({"context": policy_context, "question": req.question})
    elif intent == "hr_query":
        chain = prompt_hr | llm | parser
        answer = chain.invoke({"facts": hr_facts, "question": req.question})
    else:
        chain = prompt_hybrid | llm | parser
        answer = chain.invoke(
            {"facts": hr_facts, "context": policy_context, "question": req.question}
        )

    # Add computed multiplier if years present and question relates to overtime
    extra: Dict[str, object] = {}
    if "overtime" in req.question.lower():
        years_val: Optional[float] = None
        if "years" in hr_facts:
            try:
                years_val = float(hr_facts["years"])  # type: ignore[arg-type]
            except Exception:
                years_val = None
        if years_val is None:
            # fallback heuristic not to fail silently
            prof = USER_PROFILE.get(req.user.lower()) or {}
            years_val = float(prof.get("years", 0.0))
        extra["computed_multiplier"] = f"{pick_multiplier(years_val):.2f}x"

    return {
        "intent": intent,
        "answer": answer,
        "hr_facts": hr_facts,
        "used_policy": bool(policy_context.strip()),
        **extra,
    }


def build():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("21_hr_policy_server:app", host="0.0.0.0", port=8021, reload=False)
