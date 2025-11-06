"""
INTERVIEW STYLE Q&A:

Q: What is zero-shot prompting?
A: Zero-shot prompting means giving the LLM a task without any examples. You rely entirely
   on the model's pre-trained knowledge and your instructions. It's simple but may not match
   your desired style or format.

Q: What is few-shot prompting?
A: Few-shot prompting includes examples in your prompt to show the model the desired format,
   style, or approach. The model learns from these examples and mimics the pattern. It's
   more reliable for specific formats but uses more tokens.

Q: When should you use zero-shot vs few-shot?
A: Use zero-shot for: simple tasks, when you trust the model's default behavior, or when
   token usage matters. Use few-shot for: specific formats, consistent style requirements,
   or when you need the model to follow a particular pattern.

Q: How do you implement few-shot prompting in LangChain?
A: Include examples in your prompt template, showing input-output pairs. The model sees these
   examples and learns the pattern before processing the actual task.

SAMPLE CODE:
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI


# Q: How do you implement zero-shot prompting?
# A: Create a prompt with just instructions and the task - no examples
#    Simple instruction-only prompt (zero-shot)
def zero_shot(task: str) -> str:
    # Q: How do you structure a zero-shot prompt?
    # A: Include clear instructions and the task variable - rely on the model's training
    prompt = PromptTemplate.from_template(
        "You are concise. Perform the task and keep the answer under 6 sentences.\n\nTask: {task}"
    )
    # Q: Why set temperature=0 for zero-shot?
    # A: Lower temperature (0) makes responses more deterministic and consistent
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"task": task})


# Q: How do you implement few-shot prompting?
# A: Include example input-output pairs in your prompt to guide the model's behavior
#    Add 3 exemplars to guide style/format (few-shot)
def few_shot(task: str) -> str:
    # Q: How do you structure few-shot examples?
    # A: Create example pairs showing the desired format - the model learns from these patterns
    examples = (
        "Example 1:\nInput: Summarize benefits of TDD\nOutput: TDD encourages small, testable units...\n\n"
        "Example 2:\nInput: Give 3 downsides of long functions\nOutput: They are harder to test...\n\n"
        "Example 3:\nInput: Write a short plan to learn SQL\nOutput: Start with SELECT...\n\n"
    )
    # Q: How do you include examples in the prompt?
    # A: Add an {examples} variable in your template, then include the examples when invoking
    template = "Follow the style of the examples. Be concise.\n\n{examples}\nInput: {task}\nOutput:"
    prompt = PromptTemplate.from_template(template)
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0
    )
    chain = prompt | llm | StrOutputParser()
    # Q: How do you pass examples to the chain?
    # A: Include examples in the invoke() dictionary along with the task
    return chain.invoke({"examples": examples, "task": task})


if __name__ == "__main__":
    t = "Provide a brief plan to improve code review quality at a startup"
    print("=== Zero-shot ===\n", zero_shot(t))
    print("\n=== Few-shot ===\n", few_shot(t))
