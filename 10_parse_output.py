from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import FakeListLLM

prompt = PromptTemplate.from_template("Answer briefly: {question}")
llm = FakeListLLM(responses=["42"])
parser = StrOutputParser()

chain = prompt | llm | parser
print(chain.invoke({"question": "What is the meaning of life?"}))