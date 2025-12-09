# streamlit_app.py
"""
Streamlit frontend for the Youtuber RAG assistant (Task 2).
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"  # FastAPI endpoint


st.set_page_config(page_title="Youtuber RAG Chat", page_icon="🎥", layout="centered")

st.title("🎥 Youtuber RAG Assistant")
st.write("Ask questions based on your course transcripts. The assistant answers like *The Youtuber*.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": "..."}

# Display existing chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call FastAPI backend
    try:
        resp = requests.post(API_URL, json={"message": user_input}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("reply", "(No reply)")

    except Exception as e:
        answer = f"⚠️ Error contacting backend: {e}"

    # Show assistant message and store it
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
