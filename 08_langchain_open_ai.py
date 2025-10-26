# ensure the following environment variables are set
#export OPENAI_API_KEY="xxxx
#export AZURE_OPENAI_ENDPOINT="https://xxxx.openai.azure.com/"
#export OPENAI_API_VERSION="2024-12-01-preview" 
#export AZURE_OPENAI_DEPLOYMENT_NAME=xxx

from langchain_openai import AzureChatOpenAI
import os

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)
print(llm.invoke("Tell me a joke about programmers.").content)