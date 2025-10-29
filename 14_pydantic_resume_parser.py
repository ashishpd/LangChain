import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel


class Experience(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class Study(Experience):
    degree: Optional[str] = None
    university: Optional[str] = None
    country: Optional[str] = None
    grade: Optional[str] = None


class WorkExperience(Experience):
    company: str
    job_title: str


class Resume(BaseModel):
    first_name: str
    last_name: str
    linkedin_url: Optional[str] = None
    email_address: Optional[str] = None
    nationality: Optional[str] = None
    skill: Optional[str] = None
    study: Optional[Study] = None
    work_experience: Optional[WorkExperience] = None
    hobby: Optional[str] = None


pdf_file_path = "14_openresume-resume.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()
# please note that function calling is not enabled for all models!
# llm = ChatOllama(model="gemma3:270m")

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Use the modern structured output approach
structured_llm = llm.with_structured_output(Resume)

# Process each document
for doc in docs:
    result = structured_llm.invoke(
        f"Extract resume information from: {doc.page_content}"
    )
    print(result)
