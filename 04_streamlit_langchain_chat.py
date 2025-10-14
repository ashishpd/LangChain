"""
Minimal Streamlit chat UI using LangChain and OpenAI.

Requirements:
- pip install streamlit langchain langchain-openai openai
- Set OPENAI_API_KEY in your environment
"""
import streamlit as st
from langchain_openai import ChatOpenAI

st.title("LangChain OpenAI Chat")

user_input = st.text_input("Ask a question:")

if user_input:
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(user_input)
    st.write("**OpenAI response:**")
    st.write(response.content)
