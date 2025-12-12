# streamlit_app.py
"""
Streamlit frontend for the Youtuber RAG assistant (Task 2 & Task 4).
"""
import os
import requests
import streamlit as st

# for VG
import uuid


# --- Session handling for memory (VG / Task 4) ---
# Original:
# if "session_id" not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())# this is the end

# New (same logic, just a bit cleaner)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# this is the end


# --- Backend URL configuration ---

# Original local FastAPI endpoint:
# API_URL = "http://127.0.0.1:8000/chat"  # FastAPI endpoint

# Original local Azure Functions endpoint:
# API_URL = "http://127.0.0.1:7072/chat"
# or "http://localhost:7072/chat"

# For local development (Azure Functions on port 7072):
#API_URL = "http://127.0.0.1:7072/chat" change it later 

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:7072"   # local default
)
API_URL = f"{BACKEND_URL}/chat"
def post_chat(payload):
    r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()



# For cloud (Azure Function in swedencentral), uncomment this and comment out the line above:
# API_URL = "https://data-talks-ai-function-e0dhdmchdeawa8g2.swedencentral-01.azurewebsites.net/chat"



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

    # --- Call FastAPI backend (VG / Task 4) ---

    # Old version (was outside the if-block and causing issues):
    # payload = {
    #     "session_id": st.session_state.session_id,
    #     "message": user_input,
    # }
    #
    # try:
    #     resp = requests.post(API_URL, json=payload, timeout=60)
    #     resp.raise_for_status()
    #
    #     data = resp.json()
    #     answer = data.get("reply", "(No reply)")
    #     backend_history = data.get("history", [])
    #
    # except requests.exceptions.RequestException as e:
    #     answer = f"(Request failed: {e})"
    #     backend_history = [] #till here

    # New, fixed version (inside the if, with better error handling)
    payload = {
        "session_id": st.session_state.session_id,
        "message": user_input,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=60)

        if not resp.ok:
            # Show backend error details if status is not 2xx
            answer = f"⚠️ Backend error {resp.status_code}"
            backend_history = []
        else:
            data = resp.json()
            answer = data.get("reply", "(No reply)")
            backend_history = data.get("history", [])

    except requests.exceptions.RequestException as e:
        answer = f"(Request failed: {e})"
        backend_history = []

    # Show assistant message and store it
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Optional: show backend history (if your API returns it)
    if backend_history:
        with st.expander("Backend conversation history (from API)", expanded=False):
            for msg in backend_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                st.markdown(f"**{role}:** {content}")


# --- Old cloud API override moved & commented ---
# # for VG task
# # Before (local function)
# # API_URL = "http://127.0.0.1:7072/chat"
#
# # After (cloud function)
# API_URL = "https://data-talks-ai-function-e0dhdmchdeawa8g2.swedencentral-01.azurewebsites.net/chat"
