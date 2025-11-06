"""
INTERVIEW STYLE Q&A:

Q: How do you create a web-based chat interface for an LLM?
A: Use Streamlit - a Python framework that lets you build interactive web apps with
   minimal code. Combine it with LangChain to create a chat interface that connects
   to OpenAI's API.

Q: What makes Streamlit ideal for building LLM chat interfaces?
A: Streamlit provides simple UI components (text_input, buttons, write) and automatically
   handles the web server, so you can focus on the logic rather than HTML/CSS/JavaScript.

Q: How does the chat flow work in this example?
A: (1) User types a question in the text input, (2) When input is provided, the code
   creates a ChatOpenAI instance, (3) Invokes the model with the user's question,
   (4) Displays the response using st.write().

Q: What's the difference between a Streamlit app and a regular Python script?
A: Streamlit apps run continuously and update the UI reactively when user input changes.
   Regular scripts run once and exit. Streamlit re-runs the script on each interaction.

Requirements:
- pip install streamlit langchain langchain-openai openai
- Set OPENAI_API_KEY in your environment

SAMPLE CODE:
"""

import streamlit as st
from langchain_openai import ChatOpenAI

# Q: How do you set the title of a Streamlit app?
# A: Use st.title() to display a heading at the top of the page
st.title("LangChain OpenAI Chat")

# Q: How do you create a text input field in Streamlit?
# A: Use st.text_input() which creates an input box and returns the entered text
#    The variable user_input will be None if empty, or contain the text if entered
user_input = st.text_input("Ask a question:")

# Q: How do you conditionally execute code based on user input?
# A: Check if user_input has a value - Streamlit re-runs the script when input changes
#    This creates a reactive interface where the LLM is only called when user provides input
if user_input:
    # Q: Why create the LLM instance inside the if block?
    # A: This ensures the model is only instantiated when needed, and Streamlit's
    #    caching can optimize repeated calls. However, for better performance, you
    #    might want to create it outside and cache it.
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Q: How do you get the LLM response?
    # A: Call invoke() with the user's input - LangChain handles the API call
    response = llm.invoke(user_input)
    
    # Q: How do you display formatted text in Streamlit?
    # A: Use st.write() which accepts markdown formatting - **text** makes it bold
    st.write("**OpenAI response:**")
    st.write(response.content)
