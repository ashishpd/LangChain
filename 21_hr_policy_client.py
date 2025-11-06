"""
INTERVIEW STYLE Q&A:

Q: How do you build a client for an intent-routed RAG system?
A: Create a Streamlit UI that sends questions to an API that routes them based on intent
   (HR queries, policy queries, or hybrid). The client displays answers along with debug
   information showing intent classification and data sources used.

Q: What is intent routing in RAG systems?
A: Intent routing classifies questions into categories (HR, policy, hybrid) and uses
   different data sources and prompts for each. This ensures questions are answered
   using the most relevant information and appropriate processing logic.

Q: Why show debug information in the client?
A: Debug info (intent, facts used, policy snippets) helps users understand how the system
   arrived at answers. It's useful for troubleshooting, transparency, and building trust
   in the AI system's reasoning.

Q: How do you handle different response types from the API?
A: The API returns structured JSON with answer, intent, facts, and metadata. The client
   conditionally displays different sections based on what's available, providing a
   flexible UI that adapts to different response types.

SAMPLE CODE:
"""

import os

import httpx
import streamlit as st

API_BASE = os.getenv("HR21_API_BASE", "http://localhost:8021")


st.set_page_config(page_title="HR Policy Client 21", page_icon="🏢", layout="centered")
st.title("HR Policy Client 21")
st.caption("Intent-routed HR/Policy QA")

with st.sidebar:
    st.subheader("Server settings")
    api_base = st.text_input("API Base URL", value=API_BASE)
    st.caption("Set HR21_API_BASE env to default.")

col1, col2 = st.columns(2)
with col1:
    user = st.text_input("User", value="carol")
with col2:
    question = st.text_input("Question", value="what's my overtime rate?")

debug = st.checkbox("Show debug (intent, facts, policy)", value=True)

if st.button("Ask", type="primary"):
    if not user.strip() or not question.strip():
        st.error("Provide both user and question")
    else:
        try:
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(
                    f"{api_base}/ask",
                    json={"user": user.strip(), "question": question.strip()},
                )
            if resp.status_code != 200:
                st.error(f"Request failed: {resp.status_code} {resp.text}")
            else:
                data = resp.json()
                st.success("Answer")
                st.write(data.get("answer", "<no answer>"))
                if m := data.get("computed_multiplier"):
                    st.info(f"Computed multiplier: {m}")
                if debug:
                    st.divider()
                    st.write("Intent:", data.get("intent"))
                    st.write("Used policy:", data.get("used_policy"))
                    st.write("HR facts:")
                    st.json(data.get("hr_facts", {}))
        except Exception as e:
            st.exception(e)
