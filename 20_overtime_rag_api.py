"""
INTERVIEW STYLE Q&A:

Q: How do you build a production RAG API with FastAPI?
A: Combine FastAPI for the API layer, RAG for document retrieval, and external data
   sources (like HR APIs) to create a complete system. The API retrieves policy context,
   fetches user-specific data, and generates personalized answers.

Q: How do you combine RAG with external data sources?
A: (1) Use RAG to retrieve relevant policy documents, (2) Call external APIs (HR system)
   to get user-specific data, (3) Combine both in the LLM prompt to generate answers
   that are both policy-accurate and personalized.

Q: What's the benefit of separating API and client?
A: Separation allows: (1) Multiple clients (web, mobile, CLI) to use the same API,
   (2) Independent scaling of API and UI, (3) Easier testing and maintenance,
   (4) Reusability across different applications.

Q: How do you handle user-specific calculations in RAG?
A: Retrieve policy context via RAG, fetch user data from HR systems, then compute
   user-specific values (like overtime multipliers) based on both. Include all this
   information in the LLM prompt for accurate, personalized responses.

SAMPLE CODE:
"""

import os
from pathlib import Path
from typing import Dict, Optional

import httpx
from fastapi import FastAPI
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

POLICY_PDF = Path(__file__).parent / "20_policy_overtime.pdf"
PERSIST_BASE = str(Path(__file__).parent / ".chroma_overtime")


def ensure_policy_pdf() -> None:
    if POLICY_PDF.exists():
        return
    text = (
        "Company Overtime Policy\n\n"
        "- Employees with 1 year of service receive 1.25x overtime pay.\n"
        "- Employees with exactly 2 years of service receive 1.5x overtime pay.\n"
        "- Employees with more than 2 years of service receive 1.7x overtime pay.\n"
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
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
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


app = FastAPI(title="Overtime RAG API")
vectorstore = build_or_load_index()
llm = make_llm()
parser = StrOutputParser()


# Fake HR data
USER_YEARS: Dict[str, float] = {
    "alice": 0.8,
    "bob": 1.0,
    "carol": 2.0,
    "dave": 3.4,
}


class AskRequest(BaseModel):
    user: str
    question: str


@app.get("/hr/years/{user}")
async def hr_years(user: str) -> Dict[str, float]:
    return {"user": user, "years": USER_YEARS.get(user.lower(), 0.0)}


def pick_multiplier(years: float) -> float:
    if years > 2:
        return 1.7
    if years == 2:
        return 1.5
    if years >= 1:
        return 1.25
    return 1.0


@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, str]:
    # 1) Retrieve policy context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = await retriever.ainvoke(req.question)
    context = "\n\n".join(d.page_content for d in docs)

    # 2) Call fictitious HR API to fetch years of service
    years: Optional[float] = None
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://localhost:8000/hr/years/{req.user}")
            if resp.status_code == 200:
                years = float(resp.json().get("years", 0.0))
    except Exception:
        years = USER_YEARS.get(req.user.lower(), 0.0)
    if years is None:
        years = USER_YEARS.get(req.user.lower(), 0.0)

    # 3) Compute policy multiplier and compose answer with LLM
    multiplier = pick_multiplier(years)
    prompt = PromptTemplate.from_template(
        "You are an HR assistant. Use the policy context and years of service to answer.\n\n"
        "Policy Context:\n{context}\n\n"
        "User: {user}\nYears of service: {years}\n\n"
        "Question: {question}\n\n"
        "Answer in one short paragraph and cite the multiplier explicitly (e.g., 1.25x)."
    )
    chain = prompt | llm | parser
    answer = chain.invoke(
        {
            "context": context,
            "user": req.user,
            "years": years,
            "question": req.question,
        }
    )

    return {
        "answer": answer,
        "computed_multiplier": f"{multiplier:.2f}x",
        "years": str(years),
    }


def build():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("20_overtime_rag_api:app", host="0.0.0.0", port=8000, reload=False)
