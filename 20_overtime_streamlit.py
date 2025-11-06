"""
INTERVIEW STYLE Q&A:

Q: How do you create a client application for a RAG API?
A: Use Streamlit to build a simple web UI that calls your FastAPI RAG endpoint.
   The client sends user questions to the API and displays the responses, creating
   a user-friendly interface without implementing RAG logic in the client.

Q: Why separate the client from the API?
A: Separation enables: (1) Multiple client types (web, mobile, CLI) using the same API,
   (2) Independent updates to UI and backend, (3) Easier testing (test API separately),
   (4) Better architecture (API can serve multiple clients).

Q: How do you call a FastAPI endpoint from Streamlit?
A: Use httpx (or requests) to make HTTP POST/GET requests to your API endpoints.
   Send JSON data in the request body, handle responses, and display results in
   the Streamlit UI.

Q: What's the pattern for API client applications?
A: (1) User enters input in UI, (2) Client sends HTTP request to API with data,
   (3) API processes request and returns JSON response, (4) Client displays response
   in the UI. This is a standard REST API pattern.

SAMPLE CODE:
"""

import os

import httpx
import streamlit as st

# Q: How do you configure the API endpoint?
# A: Use environment variables with a default fallback - allows easy configuration
API_BASE = os.getenv("OVERTIME_API_BASE", "http://localhost:8000")


# Q: How do you set up a Streamlit page?
# A: Use set_page_config() for page metadata, then create UI elements
st.set_page_config(page_title="Overtime RAG Client", page_icon="🕒", layout="centered")
st.title("Overtime RAG Client")
st.caption("Ask: what's my overtime rate?")

# Q: How do you create a sidebar for settings?
# A: Use st.sidebar context manager - elements inside appear in the sidebar
with st.sidebar:
    st.subheader("Server settings")
    api_base = st.text_input("API Base URL", value=API_BASE)

# Q: How do you create input fields?
# A: Use st.text_input() for text inputs, st.columns() for side-by-side layout
st.write("Enter your username and ask your question.")
col1, col2 = st.columns(2)
with col1:
    user = st.text_input("User", value="carol")
with col2:
    question = st.text_input("Question", value="what's my overtime rate?")

# Q: How do you make API calls from Streamlit?
# A: Use httpx.Client() to make HTTP requests, handle responses, and display results
if st.button("Ask", type="primary"):
    if not user.strip() or not question.strip():
        st.error("Please provide both user and question.")
    else:
        try:
            # Q: How do you send a POST request to the API?
            # A: Use client.post() with the endpoint URL and JSON data
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{api_base}/ask",
                    json={"user": user.strip(), "question": question.strip()},
                )
                # Q: How do you handle API errors?
                # A: Check status_code - if not 200, display error message
                if resp.status_code != 200:
                    st.error(f"Request failed: {resp.status_code} {resp.text}")
                else:
                    # Q: How do you display API responses?
                    # A: Parse JSON response and display using st.write(), st.success(), etc.
                    data = resp.json()
                    st.success("Answer")
                    st.write(data.get("answer", "<no answer>"))
                    st.info(
                        f"Computed multiplier: {data.get('computed_multiplier', '<n/a>')}  |  Years: {data.get('years', '<n/a>')}"
                    )
        except Exception as e:
            st.exception(e)

st.divider()
st.subheader("Check HR Years (optional)")
check_user = st.text_input("User to check", value=user)
if st.button("Get years from HR"):
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{api_base}/hr/years/{check_user.strip()}")
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.error(f"HR request failed: {resp.status_code} {resp.text}")
    except Exception as e:
        st.exception(e)

st.caption("Set OVERTIME_API_BASE to override the API base URL.")
