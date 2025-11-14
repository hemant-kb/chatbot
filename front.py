# app.py
"""Streamlit frontend for the LangGraph PDF-only chatbot."""

import os
import tempfile
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from back import (
    chatbot,
    delete_thread,
    generate_title_from_conversation,
    get_streaming_response,
    get_thread_filename,
    get_thread_title,
    process_uploaded_file,
    retrieve_all_threads,
    save_message_to_graph,
    save_thread_title,
)

# ----------------------------------------------------------------------------- #
# Page config and custom CSS
# ----------------------------------------------------------------------------- #
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] { width: 320px; min-width: 320px; max-width: 380px; }
    [data-testid="stSidebar"] { transition: all 0.3s ease-in-out; }
    [data-testid="stSidebarContent"] { padding-top: 1rem; overflow-y: auto; }
    .file-indicator { background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0b1220 100%); border-left: 4px solid #22d3ee; box-shadow: 0 6px 20px rgba(2, 6, 23, 0.35); border-radius: 14px; padding: 14px 16px; margin: 14px 0 18px 0; color: #e2e8f0; line-height: 1.4; }
    .file-indicator strong { color: #ffffff; font-weight: 600; }
    .file-indicator small { color: #cbd5e1; font-size: 0.9rem; }
    .file-indicator .fi-row { display: flex; align-items: center; gap: 10px; }
    [data-testid="stPopover"] { margin-bottom: 0; }
    div.block-container { padding-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- #
# Small helpers
# ----------------------------------------------------------------------------- #
def generate_thread_id() -> str:
    return str(uuid.uuid4())

def add_thread(thread_id: str):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id: str) -> list[dict]:
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", []) or []
    except Exception:
        messages = []
    chat_history = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = getattr(msg, "content", "")
        if content:
            chat_history.append({"role": role, "content": content})
    return chat_history

def switch_thread(thread_id: str):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)
    st.rerun()

def reset_chat():
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    add_thread(new_id)
    save_thread_title(new_id, "New Conversation")
    st.rerun()

# ----------------------------------------------------------------------------- #
# Session state initialization
# ----------------------------------------------------------------------------- #
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

active = st.session_state.get("thread_id")
if st.session_state["chat_threads"]:
    if active not in st.session_state["chat_threads"]:
        st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
else:
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["chat_threads"] = [new_id]
    save_thread_title(new_id, "New Conversation")

st.session_state.setdefault("message_history", load_conversation(st.session_state["thread_id"]))
st.session_state.setdefault("pending_delete_id", None)
st.session_state.setdefault("pending_delete_title", "")
st.session_state.setdefault("editing_thread_id", None)
st.session_state.setdefault("uploaded_once", {})

# ----------------------------------------------------------------------------- #
# Sidebar: threads, new, rename, delete
# ----------------------------------------------------------------------------- #
st.sidebar.title("🤖 LangGraph Chatbot")

# Confirm deletion UI
if st.session_state["pending_delete_id"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Confirm Deletion")
    st.sidebar.warning(f"Delete **{st.session_state['pending_delete_title']}**?", icon="⚠️")
    dc1, dc2 = st.sidebar.columns(2)
    with dc1:
        if st.button("Yes, delete", key="confirm_delete", type="primary", use_container_width=True):
            tid = st.session_state["pending_delete_id"]
            delete_thread(tid)
            if tid in st.session_state["chat_threads"]:
                st.session_state["chat_threads"].remove(tid)
            if st.session_state.get("thread_id") == tid:
                if st.session_state["chat_threads"]:
                    st.session_state["thread_id"] = st.session_state["chat_threads"][-1]
                    st.session_state["message_history"] = load_conversation(st.session_state["thread_id"])
                else:
                    new_id = generate_thread_id()
                    st.session_state["thread_id"] = new_id
                    st.session_state["chat_threads"] = [new_id]
                    save_thread_title(new_id, "New Conversation")
                    st.session_state["message_history"] = []
            st.session_state["pending_delete_id"] = None
            st.session_state["pending_delete_title"] = ""
            st.rerun()
    with dc2:
        if st.button("Cancel", key="cancel_delete", use_container_width=True):
            st.session_state["pending_delete_id"] = None
            st.session_state["pending_delete_title"] = ""
            st.rerun()

if not st.session_state["pending_delete_id"]:
    st.sidebar.markdown("---")

if st.sidebar.button("➕ New Chat", key="new_chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.header("💬 Conversations")

if st.session_state["chat_threads"]:
    for thread_id in reversed(st.session_state["chat_threads"]):
        title = get_thread_title(thread_id) or "New Conversation"
        if st.session_state["editing_thread_id"] == thread_id:
            edit_col1, edit_col2, edit_col3 = st.sidebar.columns([5, 1, 1])
            with edit_col1:
                new_title = st.text_input(
                    "Edit title",
                    value=title,
                    max_chars=50,
                    key=f"edit_input_{thread_id}",
                    label_visibility="collapsed",
                    placeholder="Enter title (max 50 chars)",
                )
            with edit_col2:
                if st.button("✓", key=f"save_{thread_id}", help="Save"):
                    if new_title and new_title.strip():
                        save_thread_title(thread_id, new_title.strip())
                        st.session_state["editing_thread_id"] = None
                        st.rerun()
            with edit_col3:
                if st.button("✗", key=f"cancel_edit_{thread_id}", help="Cancel"):
                    st.session_state["editing_thread_id"] = None
                    st.rerun()
        else:
            col1, col2, col3 = st.sidebar.columns([4, 1, 1], vertical_alignment="center")
            with col1:
                is_active = thread_id == st.session_state["thread_id"]
                btn_type = "primary" if is_active else "secondary"
                if st.button(title, key=f"thread_{thread_id}", use_container_width=True, type=btn_type):
                    if thread_id != st.session_state["thread_id"]:
                        switch_thread(thread_id)
            with col2:
                if st.button("✏️", key=f"edit_{thread_id}", help="Edit title"):
                    st.session_state["editing_thread_id"] = thread_id
                    st.rerun()
            with col3:
                if st.button("❌", key=f"delete_{thread_id}", help="Delete conversation"):
                    st.session_state["pending_delete_id"] = thread_id
                    st.session_state["pending_delete_title"] = title
                    st.rerun()
else:
    st.sidebar.info("No conversations yet. Start chatting!")

# ----------------------------------------------------------------------------- #
# Main chat area
# ----------------------------------------------------------------------------- #
current_title = get_thread_title(st.session_state["thread_id"])
st.title(current_title)

current_file = get_thread_filename(st.session_state["thread_id"])
if current_file:
    st.markdown(
        f"""
        <div class="file-indicator">
          <div class="fi-row">
            <div>📎</div>
            <div>
              <div><strong>Document loaded:</strong> {current_file}</div>
              <div><small>Ask me anything about this document.</small></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------------------------------------------------------- #
# File upload popover (PDF only)
# ----------------------------------------------------------------------------- #
popover_col, _ = st.columns([1, 5])
with popover_col:
    with st.popover("📎 Attach PDF", use_container_width=False):
        st.caption("Upload PDF file (scanned PDFs are NOT supported)")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key=f"file_uploader_{st.session_state['thread_id']}",
            label_visibility="collapsed",
            help="Attach a PDF (text-based PDF). Scanned images without embedded text are not supported.",
        )

user_input = st.chat_input("💬 Type your message here...")

# ----------------------------------------------------------------------------- #
# Process file upload
# ----------------------------------------------------------------------------- #
if uploaded_file is not None:
    existing_file = get_thread_filename(st.session_state["thread_id"])
    already = st.session_state["uploaded_once"].get(st.session_state["thread_id"])
    if (existing_file != uploaded_file.name) or (already != uploaded_file.name):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            file_type = uploaded_file.name.split(".")[-1].lower()
            try:
                num_chunks = process_uploaded_file(
                    tmp_path,
                    file_type,
                    st.session_state["thread_id"],
                    uploaded_file.name,
                )
                if num_chunks <= 0:
                    st.error("No content extracted. The PDF may be empty or scanned without text.")
                    st.stop()
                st.session_state["message_history"].append(
                    {
                        "role": "assistant",
                        "content": (
                            f"✅ Successfully processed **{uploaded_file.name}** "
                            f"({num_chunks} chunks). Ask me anything about it!"
                        ),
                    }
                )
                st.success(f"Processed {num_chunks} chunks from {uploaded_file.name}")
                st.session_state["uploaded_once"][st.session_state["thread_id"]] = uploaded_file.name
                st.rerun()
            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ----------------------------------------------------------------------------- #
# Chat input & true streaming response
# ----------------------------------------------------------------------------- #
if user_input:
    current_thread = st.session_state["thread_id"]
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        token_stream, needs_rag = get_streaming_response(user_input, current_thread)
        for token in token_stream:
            full_response += token
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        ai_message = full_response
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        save_message_to_graph(current_thread, user_input, ai_message)

        current_title = get_thread_title(current_thread)
        if current_title == "New Conversation":
            first_msgs = [HumanMessage(content=user_input), AIMessage(content=ai_message)]
            title = generate_title_from_conversation(first_msgs)
            save_thread_title(current_thread, title)

        st.rerun()

# ----------------------------------------------------------------------------- #
# Footer
# ----------------------------------------------------------------------------- #
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
      <p>Powered by LangGraph & Azure OpenAI</p>
      <p>📎 Supports PDF files only (text-based PDFs)</p>
    </div>
    """,
    unsafe_allow_html=True,
)
