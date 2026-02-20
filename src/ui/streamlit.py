import os
import sys
import streamlit as st
import uuid
from dotenv import load_dotenv

# Ensure the repo root is on sys.path so `src` is importable
# (needed on Streamlit Cloud where cwd is the repo root but it isn't on the path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agent.graph import create_graph_with_persistence
from src.persistance.checkpointer import (
    get_checkpointer,
    list_all_threads,
    clear_thread,
    get_conversation_state,
)

load_dotenv()  # picks up .env locally; no-op in production

st.set_page_config(page_title="Research Agent", layout="wide")


# ── Cached resources (keyed by api keys so each user gets their own graph) ──
@st.cache_resource
def load_checkpointer():
    return get_checkpointer()

@st.cache_resource(show_spinner=False)
def load_graph(groq_key: str, tavily_key: str):
    """One compiled graph per unique (groq_key, tavily_key) pair."""
    return create_graph_with_persistence(
        groq_api_key=groq_key,
        tavily_api_key=tavily_key,
    )


# ── Session state defaults ────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []


# ── Load thread history from SQLite ───────────────────────────
def load_thread_history(thread_id: str, checkpointer):
    st.session_state.display_messages = []
    state = get_conversation_state(thread_id, checkpointer)
    if not state:
        return
    messages = state.get("channel_values", {}).get("messages", [])
    for msg in messages:
        if not hasattr(msg, "type"):
            continue
        if msg.type == "human" and msg.content:
            st.session_state.display_messages.append(
                {"role": "user", "content": msg.content, "tool_info": None}
            )
        elif msg.type == "ai" and msg.content:
            tool_calls = getattr(msg, "tool_calls", [])
            tool_info = [tc["name"] for tc in tool_calls] if tool_calls else None
            st.session_state.display_messages.append(
                {"role": "assistant", "content": msg.content, "tool_info": tool_info}
            )


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research Agent")
    st.caption("Powered by LangGraph + Groq")
    st.divider()

    # ── API Key inputs ────────────────────────────────────────
    st.subheader("🔑 API Keys")
    st.caption(
        "Locally these are loaded from your `.env` automatically. "
        "On the deployed app, paste your own keys here — they are never stored."
    )

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )
    tavily_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        placeholder="tvly-...",
        help="Free key at tavily.com",
    )

    keys_ok = bool(groq_key and tavily_key)

    st.divider()

    # Only show conversation controls when keys are present
    if keys_ok:
        if st.button("➕ New Conversation", use_container_width=True, type="primary"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.display_messages = []
            st.rerun()

        st.divider()
        st.subheader("Past Conversations")

        checkpointer = load_checkpointer()
        threads = list_all_threads(checkpointer)

        if not threads:
            st.caption("No past conversations yet.")
        else:
            for thread in threads:
                tid = thread["thread_id"]
                short_id = tid[-8:]
                is_active = tid == st.session_state.thread_id

                col1, col2 = st.columns([4, 1])
                with col1:
                    label = f"{'▶ ' if is_active else ''}...{short_id}"
                    if st.button(label, key=f"switch_{tid}", use_container_width=True):
                        st.session_state.thread_id = tid
                        load_thread_history(tid, checkpointer)
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{tid}", help="Delete this conversation"):
                        clear_thread(tid, checkpointer)
                        if st.session_state.thread_id == tid:
                            st.session_state.thread_id = str(uuid.uuid4())
                            st.session_state.display_messages = []
                        st.rerun()

        st.divider()
        st.caption(f"Active: ...{st.session_state.thread_id[-8:]}")


# ── MAIN CHAT AREA ────────────────────────────────────────────
st.title("Research Assistant")

if not keys_ok:
    st.warning(
        "⬅ Enter your **Groq** and **Tavily** API keys in the sidebar to start chatting.\n\n"
        "- **Groq** (free): [console.groq.com](https://console.groq.com)\n"
        "- **Tavily** (free tier): [tavily.com](https://tavily.com)"
    )
    st.stop()   # stop only when keys are genuinely missing (never happens locally with .env)


# Keys are present — lazy-initialize resources
checkpointer = load_checkpointer()

with st.spinner("Loading agent..."):
    graph = load_graph(groq_key, tavily_key)

# Load history for the active thread on first visit
if not st.session_state.display_messages:
    load_thread_history(st.session_state.thread_id, checkpointer)

# Render existing messages
for entry in st.session_state.display_messages:
    with st.chat_message(entry["role"]):
        if entry.get("tool_info"):
            with st.expander(f"🔧 Tools used: {', '.join(entry['tool_info'])}"):
                st.caption("Agent called these tools to answer your question.")
        st.markdown(entry["content"])

if prompt := st.chat_input("Ask me anything..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.display_messages.append(
        {"role": "user", "content": prompt, "tool_info": None}
    )

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        tool_calls_seen = []
        response_placeholder = st.empty()
        final_content = ""

        for event in graph.stream(
            {"messages": [("user", prompt)]},
            config,
            stream_mode="values",
        ):
            if "messages" not in event:
                continue
            last_msg = event["messages"][-1]
            if not hasattr(last_msg, "type"):
                continue

            if last_msg.type == "ai" and getattr(last_msg, "tool_calls", []):
                for tc in last_msg.tool_calls:
                    if tc["name"] not in tool_calls_seen:
                        tool_calls_seen.append(tc["name"])
                        response_placeholder.caption(
                            f"🔧 Calling: {', '.join(tool_calls_seen)}..."
                        )

            elif last_msg.type == "ai" and last_msg.content:
                final_content = last_msg.content
                response_placeholder.markdown(final_content)

        if tool_calls_seen:
            with st.expander(f"🔧 Tools used: {', '.join(tool_calls_seen)}"):
                st.caption("Agent called these tools to answer your question.")

        if final_content:
            st.session_state.display_messages.append({
                "role": "assistant",
                "content": final_content,
                "tool_info": tool_calls_seen if tool_calls_seen else None,
            })