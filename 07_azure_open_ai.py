import os

from openai import AzureOpenAI

deployment = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["OPENAI_API_KEY"],
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Tell me a joke.",
        }
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=deployment,
)

print(response.choices[0].message.content)
