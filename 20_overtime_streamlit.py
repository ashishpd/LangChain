import os

import httpx
import streamlit as st


API_BASE = os.getenv("OVERTIME_API_BASE", "http://localhost:8000")


st.set_page_config(page_title="Overtime RAG Client", page_icon="🕒", layout="centered")
st.title("Overtime RAG Client")
st.caption("Ask: what's my overtime rate?")

with st.sidebar:
    st.subheader("Server settings")
    api_base = st.text_input("API Base URL", value=API_BASE)

st.write("Enter your username and ask your question.")
col1, col2 = st.columns(2)
with col1:
    user = st.text_input("User", value="carol")
with col2:
    question = st.text_input("Question", value="what's my overtime rate?")

if st.button("Ask", type="primary"):
    if not user.strip() or not question.strip():
        st.error("Please provide both user and question.")
    else:
        try:
            with httpx.Client(timeout=30.0) as client:
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


