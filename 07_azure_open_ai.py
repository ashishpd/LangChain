"""
INTERVIEW STYLE Q&A:

Q: How do you use Azure OpenAI instead of the standard OpenAI API?
A: Use AzureOpenAI client instead of OpenAI client, and provide Azure-specific
   configuration: endpoint URL, API version, and deployment name (instead of model name).

Q: What's the difference between OpenAI and Azure OpenAI?
A: Azure OpenAI is Microsoft's hosted version of OpenAI models, providing enterprise
   features like data residency, compliance, and integration with Azure services.
   The API is similar but requires different configuration (endpoint, deployment name).

Q: What environment variables are needed for Azure OpenAI?
A: You need: (1) AZURE_OPENAI_ENDPOINT - your Azure resource endpoint URL,
   (2) OPENAI_API_KEY - your Azure API key, (3) OPENAI_API_VERSION - the API version
   to use, (4) AZURE_OPENAI_DEPLOYMENT_NAME - the name of your deployed model.

Q: Why use Azure OpenAI over standard OpenAI?
A: Benefits include: enterprise security/compliance, data residency requirements,
   integration with Azure ecosystem, and potentially better pricing for enterprise customers.

SAMPLE CODE:
"""

import os

from openai import AzureOpenAI

# Q: How do you get the Azure deployment name?
# A: Read it from environment variables - Azure uses deployment names instead of
#    model names because you can have multiple deployments of the same model
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

# Q: How do you create an Azure OpenAI client?
# A: Use AzureOpenAI() instead of OpenAI(), and provide:
#    - api_version: The Azure OpenAI API version (e.g., "2024-12-01-preview")
#    - azure_endpoint: Your Azure resource endpoint URL
#    - api_key: Your Azure API key (same env var name as OpenAI for compatibility)
client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
)

# Q: How do you make a chat completion request with Azure OpenAI?
# A: Same structure as OpenAI API, but use the deployment name as the "model" parameter
#    Additional parameters like temperature and top_p control the response randomness
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me a joke.",
        }
    ],
    max_tokens=4096,  # Maximum tokens in the response
    temperature=1.0,  # Controls randomness (0.0 = deterministic, 1.0 = more creative)
    top_p=1.0,        # Nucleus sampling parameter
    model=deployment, # Use deployment name instead of model name
)

# Q: How do you extract the response content?
# A: Same as OpenAI API - access response.choices[0].message.content
print(response.choices[0].message.content)
