"""
INTERVIEW STYLE Q&A:

Q: What is the Tree of Thought prompting technique?
A: Tree of Thought is a multi-step reasoning approach where the LLM explores multiple
   solution paths, evaluates them, deepens analysis, and then ranks them. It's like
   a structured brainstorming session that helps the model think more systematically.

Q: How does Tree of Thought differ from Chain of Thought?
A: Chain of Thought shows step-by-step reasoning for one solution. Tree of Thought
   explores multiple solutions in parallel, evaluates each, then selects the best.
   It's more comprehensive but uses more API calls.

Q: Why use a multi-step approach instead of a single prompt?
A: Complex problems benefit from breaking into steps: (1) Generate options, (2) Evaluate,
   (3) Deepen analysis, (4) Rank. Each step builds on the previous, leading to better
   final decisions than asking for everything at once.

Q: How do you chain multiple LLM calls together?
A: Use LCEL (LangChain Expression Language) with the pipe operator. Each step's output
   becomes the next step's input. The pattern is: prompt | llm | parser, and you can
   invoke chains sequentially, passing results between them.

SAMPLE CODE:
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

# Q: How do you set up the LLM for multi-step reasoning?
# A: Create your LLM instance - it will be used in each step of the tree of thought process
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Q: Why use an output parser?
# A: StrOutputParser extracts clean text from message objects, making it easy to pass
#    results between steps as strings
parser = StrOutputParser()

# Q: How do you design the first step of Tree of Thought?
# A: Create a prompt that asks for multiple solution options - this is the "brainstorming" phase
#    Step 1: Generate diverse solutions considering various factors
template = """
Step1 :
 
I have a problem related to {input}. Could you brainstorm three distinct solutions? Please consider a variety of factors such as {perfect_factors}
A:
"""

prompt1 = PromptTemplate(
    input_variables=["input", "perfect_factors"], template=template
)

# Q: How do you design the evaluation step?
# A: Create a prompt that takes the solutions from step 1 and asks for detailed evaluation
#    Step 2: Evaluate each solution's pros, cons, effort, difficulty, and success probability
template = """
Step 2:

For each of the three proposed solutions, evaluate their potential. Consider their pros and cons, initial effort needed, implementation difficulty, potential challenges, and the expected outcomes. Assign a probability of success and a confidence level to each option based on these factors

{solutions}

A:"""

prompt2 = PromptTemplate(input_variables=["solutions"], template=template)

# Q: How do you deepen the analysis?
# A: Create a prompt that takes evaluations and asks for deeper scenario planning
#    Step 3: Explore implementation details, scenarios, resources needed, and risk mitigation
template = """
Step 3:

For each solution, deepen the thought process. Generate potential scenarios, strategies for implementation, any necessary partnerships or resources, and how potential obstacles might be overcome. Also, consider any potential unexpected outcomes and how they might be handled.

{review}

A:"""

prompt3 = PromptTemplate(input_variables=["review"], template=template)

# Q: How do you finalize the decision?
# A: Create a prompt that takes the deepened analysis and asks for final ranking
#    Step 4: Rank solutions based on all previous analysis and provide justifications
template = """
Step 4:

Based on the evaluations and scenarios, rank the solutions in order of promise. Provide a justification for each ranking and offer any final thoughts or considerations for each solution
{deepen_thought_process}

A:"""

prompt4 = PromptTemplate(input_variables=["deepen_thought_process"], template=template)


# Q: How do you execute the Tree of Thought process?
# A: Chain the steps sequentially - each step's output becomes the next step's input
#    Use LCEL chains (prompt | llm | parser) for each step, then invoke with appropriate inputs
def run_tree_of_thought(user_input: str, factors: str) -> str:
    # Step 1: Generate multiple solutions
    # Q: How do you invoke the first chain?
    # A: Call invoke() with a dict containing values for all template variables
    #    The chain: fills template → sends to LLM → parses response → returns string
    solutions = (prompt1 | llm | parser).invoke(
        {
            "input": user_input,
            "perfect_factors": factors,
        }
    )

    # Step 2: Evaluate the solutions
    # Q: How do you pass results between steps?
    # A: Use the output from step 1 as input to step 2 - solutions becomes the {solutions} variable
    review = (prompt2 | llm | parser).invoke(
        {
            "solutions": solutions,
        }
    )

    # Step 3: Deepen the analysis
    # Q: How does the chain continue?
    # A: Each step builds on the previous - review (from step 2) becomes input to step 3
    deepen = (prompt3 | llm | parser).invoke(
        {
            "review": review,
        }
    )

    # Step 4: Rank and finalize
    # Q: How do you get the final result?
    # A: The last step takes all previous analysis and produces the final ranked recommendations
    ranked = (prompt4 | llm | parser).invoke(
        {
            "deepen_thought_process": deepen,
        }
    )

    return ranked


if __name__ == "__main__":
    result = run_tree_of_thought(
        user_input="choosing a project management tool for a small startup",
        factors="cost, onboarding time, integrations, AI features, mobile support",
    )
    print(result)
