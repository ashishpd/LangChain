import os

import httpx
import streamlit as st

API_BASE = os.getenv("AUTH22_API_BASE", "http://localhost:8222")


st.set_page_config(
    page_title="Auth HR/Policy Client 22", page_icon="🔐", layout="centered"
)
st.title("Auth HR/Policy Client 22")
st.caption("Login, then ask HR/Policy questions")

with st.sidebar:
    st.subheader("Server settings")
    api_base = st.text_input("API Base URL", value=API_BASE)
    roles = st.multiselect(
        "Roles", ["employee", "manager", "hr", "admin"], default=["employee"]
    )

if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = ""

st.subheader("Login")
col1, col2 = st.columns(2)
with col1:
    user = st.text_input("Username", value=st.session_state.user or "carol")
with col2:
    if st.button("Login"):
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{api_base}/auth/dev_login",
                    json={"user": user.strip(), "roles": roles},
                )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.token = data.get("access_token")
                st.session_state.user = user.strip()
                st.success("Logged in")
            else:
                st.error(resp.text)
        except Exception as e:
            st.exception(e)

st.divider()
st.subheader("Ask a question")
q = st.text_input("Question", value="what's my overtime rate?")
if st.button("Ask", type="primary"):
    if not st.session_state.token:
        st.error("Please login first")
    else:
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            with httpx.Client(timeout=45.0, headers=headers) as client:
                resp = client.post(
                    f"{api_base}/ask",
                    json={"user": st.session_state.user, "question": q},
                )
            if resp.status_code == 200:
                data = resp.json()
                st.success("Answer")
                st.write(data.get("answer"))
                st.caption(
                    f"Policy used: {data.get('policy_used')} | HR facts: {list((data.get('facts') or {}).keys())}"
                )
                if ot := data.get("overtime"):
                    st.info(f"Overtime: {ot}")
            else:
                st.error(f"{resp.status_code} {resp.text}")
        except Exception as e:
            st.exception(e)

st.caption("Set AUTH22_API_BASE to point to a different server URL.")
