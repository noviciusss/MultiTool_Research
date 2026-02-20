import streamlit as st 
import uuid

from src.agent.graph import create_graph_with_persistence
from src.persistance.checkpointer import (
    get_checkpointer,
    list_all_threads,
    clear_thread,
    get_conversation_state
)
#Page config 
st.set_page_config(
    page_title="Research Agent",
    layout="wide",
    
)

# ----------------Cached Resorces ----------------
# why st.cache_resource:Streamlut reruns the entire script on every user interaction.
#without caching graph and checkpointer would be re-initialized (and DB re-opened)on every keypress
@st.cache_resource
def load_graph():
    return create_graph_with_persistence()

@st.cache_resource
def load_checkpointer():
    return get_checkpointer()

graph = load_graph()
checkpointer = load_checkpointer()

# ------------------ Session State initialization ----------------
# st.session_state persists across reruns within the same browser tab 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # Unique ID for this conversation thread

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # Messages to display in the UI
    
    
# Helper : load history from sqlite into display_message 
# called when switching threads or on first load 
def load_thread_history(thread_id:str):
    """pull saved messages from Sqlite abd show at display_messages"""
    st.session_state.display_messages = []  # Clear current display messages
    state = get_conversation_state(thread_id,checkpointer)
    if not state:
        return # No history for this thread
    
    messages = state.get("channel_values",{}).get("messages",[])
    for msg in messages:
        if not hasattr(msg,"type"):
            continue
        
        if msg.type == "human" and msg.content:
            st.session_state.display_messages.append({
                "role":"user",
                "content": msg.content,
                "tool_info": None
            })
        elif msg.type == "ai" and msg.content:
            #collect tool call info if available
            tool_calls = getattr(msg,"tool_calls",[])
            tool_info = [tc["name"] for tc in tool_calls] if tool_calls else None
            st.session_state.display_messages.append({
                "role":"assistant",
                "content": msg.content,
                "tool_info": tool_info
            })
            
#----------------Load history for current thread on first render --------- 
if not st.session_state.display_messages:
    load_thread_history(st.session_state.thread_id)
    

#Sidebar : Thread management
with st.sidebar:
    st.title("Research Agent")
    st.caption("Powered by langgraph and Groq")
    st.rerun() 
    
st.divider()
st.subheader("Conversations...")

threads = list_all_threads(checkpointer)

if not threads:
    st.caption("No past coversations yet.")
else: 
    for thread in threads:
        tid = thread["thread_id"]
        count = thread["checkpoint_count"]
        #truncate uuid for display
        short_id = tid[-8:] if len(tid) > 8 else tid
        is_active = tid == st.session_state.thread_id
        
        col1,col2 = st.columns([4,1])
        with col1:
                label = f"{'▶ ' if is_active else ''}...{short_id}"
                if st.button(label, key=f"switch_{tid}", use_container_width=True):
                    st.session_state.thread_id = tid
                    load_thread_history(tid)
                    st.rerun()

        with col2:
                if st.button("🗑", key=f"del_{tid}", help="Delete this conversation"):
                    clear_thread(tid, checkpointer)
                    if st.session_state.thread_id == tid:
                        st.session_state.thread_id = str(uuid.uuid4())
                        st.session_state.display_messages = []
                    st.rerun()

    st.divider()
    st.caption(f"Active thread: ...{st.session_state.thread_id[-8:]}")

# ─────────────────────────────────────────────────────────────
# MAIN AREA — chat interface
# ─────────────────────────────────────────────────────────────
st.title("Research Assistant")

# Render existing messages
for entry in st.session_state.display_messages:
    with st.chat_message(entry["role"]):
        # Show tool usage as expandable section above the response
        if entry.get("tool_info"):
            with st.expander(f"🔧 Tools used: {', '.join(entry['tool_info'])}"):
                st.caption("Agent called these tools to research your question.")
        st.markdown(entry["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.display_messages.append({
        "role": "user",
        "content": prompt,
        "tool_info": None,
    })

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # Stream agent response
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

            # Agent is calling tools — collect names for display
            if last_msg.type == "ai" and getattr(last_msg, "tool_calls", []):
                for tc in last_msg.tool_calls:
                    if tc["name"] not in tool_calls_seen:
                        tool_calls_seen.append(tc["name"])

            # Agent has final answer
            elif last_msg.type == "ai" and last_msg.content:
                final_content = last_msg.content
                response_placeholder.markdown(final_content)

        # Show tool calls used in this response
        if tool_calls_seen:
            with st.expander(f"🔧 Tools used: {', '.join(tool_calls_seen)}"):
                st.caption("Agent called these tools to research your question.")
        
        
        # Save to display state
        if final_content:
            st.session_state.display_messages.append({
                "role": "assistant",
                "content": final_content,
                "tool_info": tool_calls_seen if tool_calls_seen else None,
            })