"""
INTERVIEW STYLE Q&A:

Q: What are LangChain tools and how do you create them?
A: Tools are functions that agents can call to perform actions (search, compute, retrieve).
   Use the @tool decorator to convert regular functions into LangChain tools. Tools
   have names, descriptions, and schemas that agents use to decide when to call them.

Q: How do tools enforce authorization?
A: Tools can check user permissions before executing. They receive caller information
   (user, roles) and validate access based on field sensitivity and user roles.
   This ensures only authorized users can access sensitive data.

Q: What's the difference between tools and regular functions?
A: Tools are structured for agent use - they have schemas, descriptions, and can be
   discovered by agents. Regular functions are just code. Tools enable agents to
   dynamically decide which actions to take based on the task.

Q: How do you structure tools for a RAG system?
A: Create separate tools for different operations: policy_retrieve (RAG search),
   hr_get (fetch HR data with auth), compute_overtime (calculations). Each tool
   handles one responsibility, making the system modular and testable.

SAMPLE CODE:
"""

import os
from pathlib import Path
from typing import Dict, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Build a reusable vectorstore over 21_policy_overtime.pdf (or create 22_policy_overtime.pdf)
PDF_PATH = Path(__file__).parent / "21_policy_overtime.pdf"
PERSIST_BASE = str(Path(__file__).parent / ".chroma_tools22")


def _make_embeddings():
    if os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"):
        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
        )
    return OpenAIEmbeddings()


def _build_vectorstore() -> Chroma:
    embeddings = _make_embeddings()
    persist_dir = (
        f"{PERSIST_BASE}_{os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT','openai')}"
    )
    if Path(persist_dir).exists():
        try:
            return Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        except Exception:
            pass
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = splitter.split_documents(pages)
    return Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=persist_dir
    )


_VECTORSTORE = _build_vectorstore()


# Minimal in-memory HR profile (server should enforce auth; tools do field checks)
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


def _authorized(
    field: str, subject_user: str, caller_user: str, roles: List[str]
) -> bool:
    # Self-access allowed for some fields; HR role can access more; deny by default
    if field in {"years", "title", "manager", "pto_balance"}:
        return (
            caller_user == subject_user
            or ("manager" in roles)
            or ("hr" in roles)
            or ("admin" in roles)
        )
    if field in {"dob", "salary"}:
        # Allow self to view their own sensitive fields; others require HR/Admin
        return (caller_user == subject_user) or ("hr" in roles) or ("admin" in roles)
    return False


@tool("policy_retrieve")
def policy_retrieve(query: str) -> dict:
    """Retrieve policy snippets relevant to the query."""
    docs = _VECTORSTORE.similarity_search(query, k=4)
    return {"snippets": [d.page_content for d in docs]}


@tool("hr_get")
def hr_get(user: str, fields: List[str], caller_user: str, roles: List[str]) -> dict:
    """Get HR fields for a user. Enforces auth based on roles and whether caller is the subject."""
    profile = USER_PROFILE.get(user.lower()) or {}
    out: Dict[str, object] = {}
    for f in fields:
        if _authorized(f, user.lower(), caller_user.lower(), roles):
            if f == "years":
                out[f] = float(profile.get("years", 0.0))
            elif f in profile:
                out[f] = profile[f]
    return out


@tool("compute_overtime")
def compute_overtime(years: float) -> dict:
    """Compute overtime multiplier from years of service."""
    if years > 2:
        mult = 1.7
    elif years == 2:
        mult = 1.5
    elif years >= 1:
        mult = 1.25
    else:
        mult = 1.0
    return {"multiplier": mult}
