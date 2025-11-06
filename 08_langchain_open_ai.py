"""
INTERVIEW STYLE Q&A:

Q: How do you use Azure OpenAI with LangChain?
A: Use AzureChatOpenAI instead of ChatOpenAI, and provide the Azure deployment name.
   LangChain automatically reads Azure configuration from environment variables.

Q: What's the advantage of using LangChain's AzureChatOpenAI?
A: It provides the same simple interface as ChatOpenAI but works with Azure OpenAI,
   and it automatically handles Azure-specific configuration from environment variables.
   Your code remains clean and provider-agnostic.

Q: What environment variables does AzureChatOpenAI use?
A: It automatically reads: AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY, OPENAI_API_VERSION,
   and AZURE_OPENAI_DEPLOYMENT_NAME. You only need to explicitly pass the deployment name
   in the constructor (or it can read it from env vars too).

Q: How does this compare to using AzureOpenAI client directly?
A: LangChain's wrapper provides a simpler interface (just invoke() with a string),
   integrates with LangChain chains and features, and maintains consistency across
   different LLM providers.

Environment Variables Required:
- export OPENAI_API_KEY="your-azure-api-key"
- export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
- export OPENAI_API_VERSION="2024-12-01-preview"
- export AZURE_OPENAI_DEPLOYMENT_NAME="your-deployment-name"

SAMPLE CODE:
"""

import os

from langchain_openai import AzureChatOpenAI

# Q: How do you create a LangChain Azure OpenAI chat model?
# A: Use AzureChatOpenAI and pass the deployment name - it automatically reads
#    other Azure configuration (endpoint, API version, API key) from environment variables
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: How do you invoke the Azure model with LangChain?
# A: Same as regular ChatOpenAI - call invoke() with a string prompt
#    The .content attribute gives you the text response
print(llm.invoke("Tell me a joke about programmers.").content)
