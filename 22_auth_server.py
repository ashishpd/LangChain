import importlib.util
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel

# Dynamically load tools from a filename that starts with digits (not a valid module name)
_TOOLS_PATH = Path(__file__).parent / "22_tools.py"
_spec = importlib.util.spec_from_file_location("tools22", str(_TOOLS_PATH))
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load tools module from 22_tools.py")
tools22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tools22)  # type: ignore[attr-defined]

APP_SECRET = os.getenv("APP_JWT_SECRET", "dev-secret-change-me")
ISSUER = "hr-demo-22"


def make_llm():
    if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
        )
    return ChatOpenAI(temperature=0, model="gpt-4o-mini")


def encode_jwt(sub: str, roles: List[str]) -> str:
    now = int(time.time())
    payload = {"iss": ISSUER, "sub": sub, "roles": roles, "iat": now, "exp": now + 3600}
    return jwt.encode(payload, APP_SECRET, algorithm="HS256")


def decode_jwt(token: str) -> Dict[str, object]:
    try:
        return jwt.decode(
            token,
            APP_SECRET,
            algorithms=["HS256"],
            options={"require": ["exp", "iat", "iss", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}")


bearer = HTTPBearer(auto_error=False)


async def auth_dep(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
) -> Dict[str, object]:
    if not creds or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing bearer token")
    claims = decode_jwt(creds.credentials)
    return claims


app = FastAPI(title="Auth HR/Policy Server 22")
llm = make_llm()
parser = StrOutputParser()


@app.post("/auth/dev_login")
async def dev_login(body: Dict[str, object]) -> Dict[str, str]:
    user = str(body.get("user", "")).strip().lower()
    if not user:
        raise HTTPException(status_code=400, detail="user required")
    roles = body.get("roles") or []
    if not isinstance(roles, list):
        raise HTTPException(status_code=400, detail="roles must be list")
    token = encode_jwt(user, roles)
    return {"access_token": token, "token_type": "bearer"}


class AskRequest(BaseModel):
    user: Optional[str] = None
    question: str


@app.post("/ask")
async def ask(
    req: AskRequest, claims: Dict[str, object] = Depends(auth_dep)
) -> Dict[str, object]:
    user = (req.user or str(claims.get("sub", ""))).strip().lower()
    question = req.question
    roles: List[str] = (
        list(claims.get("roles", [])) if isinstance(claims.get("roles"), list) else []
    )

    sys = (
        "You decide which tools to call. Use hr_get for personal/HR facts (enforces auth). "
        "Use policy_retrieve for company rules. Combine results faithfully to answer."
    )
    prompt = PromptTemplate.from_template(
        "{sys}\n\nUser: {user}\nQuestion: {question}\n\nIf overtime depends on years, you may compute via compute_overtime."  # guidance
    )

    # Simple loop: 1) retrieve policy; 2) try HR facts for common fields; 3) compute overtime if asked
    policy = tools22.policy_retrieve.invoke({"query": question})

    wanted_fields = []
    ql = question.lower()
    if "overtime" in ql and "years" not in wanted_fields:
        wanted_fields.append("years")
    if any(k in ql for k in ["manager", "title", "pto"]):
        if "manager" in ql:
            wanted_fields.append("manager")
        if "title" in ql:
            wanted_fields.append("title")
        if "pto" in ql:
            wanted_fields.append("pto_balance")

    hr = (
        tools22.hr_get.invoke(
            {
                "user": user,
                "fields": wanted_fields,
                "caller_user": str(claims.get("sub", "")).lower(),
                "roles": roles,
            }
        )
        if wanted_fields
        else {"output": {}}
    )
    facts = hr.get("output", {}) if isinstance(hr, dict) else {}

    extra = {}
    if "years" in facts and "overtime" in ql:
        ot = tools22.compute_overtime.invoke({"years": float(facts["years"])})
        extra = ot.get("output", {}) if isinstance(ot, dict) else {}

    chain = prompt | llm | parser
    answer = chain.invoke(
        {
            "sys": sys,
            "user": user,
            "question": question,
            "policy": policy,
            "facts": facts,
            "extra": extra,
        }
    )

    return {
        "answer": answer,
        "facts": facts,
        "policy_used": bool(policy.get("snippets")),
        **({"overtime": extra} if extra else {}),
    }


def build():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("22_auth_server:app", host="0.0.0.0", port=8222, reload=False)
