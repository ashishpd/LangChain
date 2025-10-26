"""
Streamlit app: Ask LLM for US GDP (last 5 years) and plot.

Requirements:
- pip install streamlit langchain langchain-openai openai pandas
- Set OPENAI_API_KEY in your environment
"""

import re

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

st.title("US GDP (Last 5 Years) via LLM")

if st.button("Get and Plot US GDP Data"):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = "Provide a table of US GDP (in trillions USD) for the last 5 years. Only give the table."
    response = llm.invoke(prompt)
    text = response.content.strip()

    # Try to extract a table (robust to CSV, markdown, or plain)
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not set(line.strip()) <= set("-|")
    ]
    data = []
    for line in lines:
        # Try to find year and GDP in each line
        match = re.findall(r"(\d{4})[^\d]*([\d,.]+)", line)
        if match:
            year, gdp = match[0]
            gdp = float(gdp.replace(",", ""))
            data.append((int(year), gdp))
    if data:
        df = pd.DataFrame(data, columns=["Year", "GDP (Trillions USD)"]).sort_values(
            "Year"
        )
        st.line_chart(df.set_index("Year"))
        st.write(df)
    else:
        st.error("Could not parse GDP data from LLM response.")
