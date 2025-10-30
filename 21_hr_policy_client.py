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
