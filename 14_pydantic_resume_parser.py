"""
INTERVIEW STYLE Q&A:

Q: How do you extract structured data from unstructured documents like PDFs?
A: Use LangChain's document loaders to load PDFs, then use structured output with Pydantic
   models to extract specific fields. The LLM is guided by the Pydantic schema to return
   data in the exact format you need.

Q: What is structured output and why is it useful?
A: Structured output forces the LLM to return data matching a specific schema (like a
   Pydantic model) instead of free-form text. This ensures consistent, parseable data
   that you can directly use in your application without manual parsing.

Q: How do Pydantic models help with structured extraction?
A: Pydantic models define the exact structure you want (fields, types, optional vs required).
   LangChain uses these models to guide the LLM's output, ensuring it returns JSON that
   matches your schema exactly.

Q: What's the advantage of using with_structured_output()?
A: It automatically handles the conversion from LLM response to your Pydantic model. Instead
   of parsing JSON strings manually, you get a typed Pydantic object directly.

SAMPLE CODE:
"""

import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

# Q: How do you define the data structure you want to extract?
# A: Create Pydantic models that represent the structure - BaseModel provides validation
#    Optional fields allow for missing data, required fields ensure they're always present

# Q: How do you model nested/complex data?
# A: Use inheritance - Study and WorkExperience inherit from Experience, adding specific fields
#    This creates a hierarchy: Experience (base) → Study/WorkExperience (specific types)
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
    company: str  # Required field - must be present
    job_title: str  # Required field


# Q: How do you define the main data structure?
# A: Create a root model that contains all the fields you want to extract
#    Use Optional for fields that might not be present in every document
class Resume(BaseModel):
    first_name: str  # Required
    last_name: str  # Required
    linkedin_url: Optional[str] = None
    email_address: Optional[str] = None
    nationality: Optional[str] = None
    skill: Optional[str] = None
    study: Optional[Study] = None  # Nested model
    work_experience: Optional[WorkExperience] = None  # Nested model
    hobby: Optional[str] = None


# Q: How do you load PDF documents?
# A: Use PyPDFLoader from langchain_community - it loads and splits PDFs into document chunks
pdf_file_path = "14_openresume-resume.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

# Q: How do you set up the LLM for structured output?
# A: Create your LLM instance, then use with_structured_output() to enable structured extraction
#    Note: Function calling/structured output requires models that support it (like GPT-4)
# please note that function calling is not enabled for all models!
# llm = ChatOllama(model="gemma3:270m")

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: How do you enable structured output?
# A: Call with_structured_output() with your Pydantic model - this configures the LLM
#    to return data matching your schema instead of free-form text
# Use the modern structured output approach
structured_llm = llm.with_structured_output(Resume)

# Q: How do you extract structured data from documents?
# A: Process each document chunk, asking the LLM to extract information matching your schema
#    The structured_llm automatically formats the response as a Pydantic Resume object
# Process each document
for doc in docs:
    # Q: How do you invoke structured extraction?
    # A: Call invoke() with a prompt asking to extract data - the LLM returns a Resume object
    #    No need to parse JSON - you get a typed Pydantic object directly
    result = structured_llm.invoke(
        f"Extract resume information from: {doc.page_content}"
    )
    print(result)
