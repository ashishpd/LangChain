"""
INTERVIEW STYLE Q&A:

Q: What's the difference between a system message and a user message in LLM prompts?
A: System messages set the context, role, or instructions for the LLM (like "You are a translator").
   User messages contain the actual content or question. System messages guide behavior,
   while user messages are what the model responds to.

Q: How do you create a prompt template with both system and user messages?
A: Use ChatPromptTemplate.from_messages() with a list of tuples. Each tuple has a role
   ("system" or "user") and a template string. Variables in templates are filled when invoked.

Q: Why use ChatPromptTemplate instead of PromptTemplate?
A: ChatPromptTemplate is designed for chat models that understand message roles (system/user/assistant).
   It properly formats messages for chat APIs, while PromptTemplate is for simple text completion.

Q: How do you use variables in both system and user templates?
A: Define variables in both templates using {variable_name}. When you invoke the template,
   provide values for all variables in a single dictionary.

SAMPLE CODE:
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Q: How do you initialize a chat model?
# A: Create an instance with the model name - ChatOllama for local models, or ChatOpenAI for OpenAI
llm = ChatOllama(model="gemma3:270m")

# Q: How do you define a system message template?
# A: Create a string with variables - this will be the system message that sets the LLM's role/context
#    The {language} variable will be filled when the template is invoked
system_template = "Translate the following from English into {language}"

# Q: How do you create a chat prompt template with multiple message types?
# A: Use ChatPromptTemplate.from_messages() with a list of (role, template) tuples
#    - ("system", ...) sets the system message/instructions
#    - ("user", ...) contains the user's input
#    Both templates can have variables that get filled together
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Q: How do you fill in the template variables?
# A: Call invoke() with a dictionary containing values for all variables
#    This creates the actual messages with variables replaced
prompt = prompt_template.invoke({"language": "Spanish", "text": "hi how are you?"})

# Q: How do you get the LLM response?
# A: Pass the formatted prompt to llm.invoke() - it returns a message object
response = llm.invoke(prompt)
print(response.content)
