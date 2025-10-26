import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

prompt = PromptTemplate.from_template("Write a motto about {topic}.")
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

chain = prompt | llm  # LCEL pipeline (LangChain Expression Language)
print(chain.invoke({"topic": "software craftsmanship"}))
