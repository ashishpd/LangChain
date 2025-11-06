"""
INTERVIEW STYLE Q&A:

Q: How can you use an LLM to generate structured data and visualize it?
A: Ask the LLM to provide data in a specific format (like a table), parse the response
   to extract structured data, convert it to a pandas DataFrame, and use Streamlit's
   charting capabilities to visualize it.

Q: What's the challenge with getting structured data from LLMs?
A: LLMs return text, not structured data. You need to parse the text response to extract
   the data you need, which requires pattern matching (regex) and error handling for
   cases where the format might vary.

Q: How does this example combine LLM capabilities with data visualization?
A: (1) LLM generates GDP data in text format, (2) Regex parsing extracts year-GDP pairs,
   (3) Data is converted to a pandas DataFrame, (4) Streamlit's line_chart() creates
   a visualization, (5) The raw data table is also displayed.

Q: Why use regex for parsing instead of expecting JSON from the LLM?
A: This example shows a robust approach that works with various text formats (CSV,
   markdown tables, plain text). For production, you might prefer asking the LLM for
   JSON and using structured output parsers.

Requirements:
- pip install streamlit langchain langchain-openai openai pandas
- Set OPENAI_API_KEY in your environment

SAMPLE CODE:
"""

import re

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI

# Q: How do you create a title for your Streamlit app?
# A: Use st.title() to display a heading
st.title("US GDP (Last 5 Years) via LLM")

# Q: How do you create a button that triggers an action?
# A: Use st.button() which returns True when clicked, causing the code inside
#    the if block to execute
if st.button("Get and Plot US GDP Data"):
    # Q: How do you get structured data from an LLM?
    # A: Ask the LLM to provide data in a specific format (table), then parse the response
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = "Provide a table of US GDP (in trillions USD) for the last 5 years. Only give the table."
    response = llm.invoke(prompt)
    text = response.content.strip()

    # Q: How do you clean up the LLM response before parsing?
    # A: Split into lines and filter out empty lines and separator lines (lines with only dashes or pipes)
    #    This makes the parsing more robust to different table formats
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not set(line.strip()) <= set("-|")
    ]
    
    # Q: How do you extract structured data from unstructured text?
    # A: Use regex (regular expressions) to find patterns - in this case, 4-digit years
    #    followed by numbers that represent GDP values
    data = []
    for line in lines:
        # Q: What does this regex pattern do?
        # A: r"(\d{4})[^\d]*([\d,.]+)" finds: (1) 4 digits (year), (2) any non-digits,
        #    (3) digits/commas/dots (GDP value). The parentheses create capture groups.
        match = re.findall(r"(\d{4})[^\d]*([\d,.]+)", line)
        if match:
            year, gdp = match[0]
            # Q: Why remove commas from the GDP value?
            # A: Commas are formatting characters - convert to float by removing them first
            gdp = float(gdp.replace(",", ""))
            data.append((int(year), gdp))
    
    # Q: How do you handle cases where parsing fails?
    # A: Check if data was successfully extracted - if the list is empty, show an error
    if data:
        # Q: How do you create a DataFrame and sort it?
        # A: Use pd.DataFrame() with the data and column names, then sort_values() by Year
        df = pd.DataFrame(data, columns=["Year", "GDP (Trillions USD)"]).sort_values(
            "Year"
        )
        # Q: How do you create a line chart in Streamlit?
        # A: Use st.line_chart() with a DataFrame where the index is the x-axis
        #    set_index("Year") makes Year the index for the chart
        st.line_chart(df.set_index("Year"))
        # Q: How do you display the raw data table?
        # A: Use st.write() to display the DataFrame as a formatted table
        st.write(df)
    else:
        # Q: How do you show error messages in Streamlit?
        # A: Use st.error() which displays a red error message to the user
        st.error("Could not parse GDP data from LLM response.")
