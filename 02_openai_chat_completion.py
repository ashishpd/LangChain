import os
from openai import OpenAI

# Ensure API key is present
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Explain AI in 2 lines"}],
)

# Access the assistant reply via the new response shape
content = response.choices[0].message.content.strip()
print(content)