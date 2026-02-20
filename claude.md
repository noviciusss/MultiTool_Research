# Claude Context: Multi-Tool Research Agent

> **Purpose:** This file provides complete project context for AI assistants (Claude, GPT, etc.) in future sessions. Read this first before touching any code.

**Last Updated:** February 20, 2026
**Current Phase:** Phase 4 Complete → Phase 5 (Deployment) is next
**Project Status:** Core agent functional, all tools working, SQLite persistence working, Streamlit UI working with user-provided API keys

---

## What Is This Project?

A conversational AI research assistant that reasons over multiple data sources using the **ReAct pattern** (Reason + Act loop). The agent decides which tools to call, calls them, sees the results, and decides whether to call more tools or answer the user.

**It can:**
- Search the web for current info (Tavily)
- Find academic papers (ArXiv)
- Look up general knowledge (Wikipedia)
- Do math and statistics (Calculator)
- Remember full conversations across sessions (SQLite)

**It cannot yet:**
- Show a UI (Streamlit file is empty - Phase 4)
- Run arbitrary Python (python_repl_tool.py is empty)
- Handle multiple users concurrently (SQLite limitation)

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent framework | LangGraph | Native support for cyclic graphs (ReAct needs loops), built-in checkpointing |
| LLM | Groq - Llama 3.3 70B Versatile | Fast (500+ tok/s), free tier, quality rivals GPT-4 |
| Persistence | SQLite via `langgraph-checkpoint-sqlite` | Zero setup, file-based, ACID compliant, easy to swap later |
| Web search | Tavily | Built for LLMs, returns clean structured results |
| Paper search | ArXiv (langchain-community) | Free, comprehensive academic database |
| General knowledge | Wikipedia (langchain-community) | Free, no API key needed |
| UI | Streamlit (not yet built) | Fastest path to chat interface |
| Env config | python-dotenv | Industry standard .env loading |

---

## Project Structure

```
MultiTool_Research/
├── .env                          # API keys (never commit!)
├── .gitignore
├── requirements.txt              # All dependencies
├── run.py                        # Entry point - EMPTY (Phase 4)
├── README.md
├── System_design.md              # Phase-by-phase design decisions
├── claude.md                     # This file
│
├── data/
│   └── checkpoints.db            # SQLite DB - auto-created on first run
│
└── src/
    ├── __init__.py
    ├── agent/
    │   ├── __init__.py
    │   └── graph.py              # CORE: ReAct graph, all agent logic lives here
    ├── tools/
    │   ├── __intit__.py          # TYPO: should be __init__.py (doesn't break anything)
    │   ├── tavily_tool.py        # Web search
    │   ├── arxiv_tool.py         # Academic papers
    │   ├── wikipedia_tool.py     # General knowledge
    │   ├── calculator_tool.py    # Math + statistics
    │   └── python_repl_tool.py   # EMPTY - not implemented yet
    ├── persistance/              # TYPO in folder name: should be "persistence"
    │   ├── __init__.py
    │   └── checkpointer.py       # SQLite checkpoint management
    └── ui/
        └── streamlit.py          # EMPTY - Phase 4
```

---

## File-by-File Reference

### `src/agent/graph.py` — The Brain

This is the most important file. Everything agent-related lives here.

**State:**
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```
`add_messages` is a LangGraph reducer — it merges new messages into existing history instead of replacing it. Without it, every turn overwrites the previous conversation.

**Key functions:**

| Function | What It Does |
|----------|--------------|
| `create_agent_node(tools)` | Factory — returns a node function with tools bound to the LLM |
| `should_continue(state)` | Conditional edge — returns `"tools"` if last message has tool_calls, else `"end"` |
| `create_graph()` | Builds and compiles graph WITHOUT persistence |
| `create_graph_with_persistence(db_path)` | Builds and compiles graph WITH SQLite checkpointer |

**LLM config:**
```python
ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
```
Temperature 0 = deterministic. Same question always gets same tool selection. Good for debugging.

**Graph flow:**
```
START → agent → should_continue → tools → agent (loop)
                              ↘ END
```

**System prompt** tells the LLM:
- Which 4 tools exist and when to use each
- To use multiple tools if needed
- To cite sources

**`__main__` block** tests both modes:
1. Without persistence — invokes on ArXiv query
2. With persistence — streams with hardcoded `thread_id = "test_conversation_1"` so you can run it twice to verify resume works

---

### `src/persistance/checkpointer.py` — Conversation Memory

**Functions:**

| Function | Purpose |
|----------|---------|
| `get_checkpointer(db_path)` | Creates `SqliteSaver` instance, ensures `data/` dir exists |
| `get_conversation_state(thread_id, checkpointer)` | Returns latest checkpoint state for a thread or `None` |
| `clear_thread(thread_id, checkpointer)` | Deletes all rows for a thread from DB |
| `list_all_threads(checkpointer)` | Returns list of `{thread_id, checkpoint_count, last_updated}` |

**Critical:** Must use `SqliteSaver.from_conn_string(path)` — NOT `SqliteSaver(path)`. The class method sets up connection pooling, creates the schema, enables WAL mode. Direct constructor expects an already-connected `sqlite3.Connection` object.

**Database schema** (auto-created by LangGraph):
```sql
CREATE TABLE checkpoints (
    thread_id TEXT,
    checkpoint_id TEXT,
    parent_id TEXT,
    checkpoint BLOB,   -- serialized state
    metadata TEXT,
    created_at TIMESTAMP
);
```

---

### `src/tools/tavily_tool.py`
```python
TavilySearch(max_results=3, search_depth='basic', include_answer=True)
```
Requires `TAVILY_API_KEY` in `.env`. Use for: current events, news, anything time-sensitive.

### `src/tools/arxiv_tool.py`
```python
ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=1500))
```
No API key needed. Uses `ArxivQueryRun` (returns string) NOT `ArxivRetriever` (returns `List[Document]` which the LLM can't read). The old retriever approach is commented out at the top of the file with explanation.

### `src/tools/wikipedia_tool.py`
```python
WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000))
```
No API key needed. Use for: definitions, historical facts, general background.

### `src/tools/calculator_tool.py`
```python
@tool
def calculator(expression: str) -> str:
    # restricted eval with safe_dict
```
Uses `eval()` with `{"__builtins__": None}` and a whitelist of math/stats functions. Safe — users cannot execute arbitrary code. Supports: `sqrt, sin, cos, tan, log, mean, median, stdev`.

---

## Environment Variables

**Required:**
```bash
GROQ_API_KEY=       # console.groq.com
TAVILY_API_KEY=     # tavily.com
```

**Optional (LangSmith tracing):**
```bash
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=multi-tool-research
LANGSMITH_TRACING_V2=true
```
ArXiv and Wikipedia need no keys.

---

## Phase Status

### Phase 1 — Core Agent ✅
- `AgentState` with `add_messages` reducer
- LLM node with Groq
- Conditional routing (`should_continue`)
- Basic graph: START → agent → END

### Phase 2 — Multiple Tools ✅
- Tavily, ArXiv, Wikipedia, Calculator all working
- ReAct loop: agent ↔ tools, iterates until satisfied
- Tool selection done by LLM (not hard-coded rules)
- `python_repl_tool.py` is still empty

### Phase 3 — Persistence ✅
- `SqliteSaver.from_conn_string()` checkpointer
- `create_graph_with_persistence()` in graph.py
- `get_checkpointer`, `clear_thread`, `list_all_threads` in checkpointer.py
- Thread IDs are UUID4 strings
- Config pattern: `{"configurable": {"thread_id": "..."}}`

### Phase 4 — Streamlit UI ✅
- Chat interface, sidebar, thread management
- `@st.cache_resource(groq_key, tavily_key)` — separate graph per user key pair
- `st.stop()` gates chat behind key entry — users provide their own API keys
- Streaming tool call progress display

### Phase 5 — Deployment ⏳ (Next)
- Target: Streamlit Community Cloud (free)
- SQLite works locally; swap to PostgreSQL (`PostgresSaver`) for cloud
- Users bring their own keys — no secrets needed in the repo

---

## Common Tasks

### Run agent (no UI)
```bash
python -m src.agent.graph
```
Runs the `__main__` block — tests ArXiv search then persistence with `thread_id = "test_conversation_1"`.

### Test a specific tool
```bash
python -m src.tools.tavily_tool
python -m src.tools.arxiv_tool
python -m src.tools.wikipedia_tool
python -m src.tools.calculator_tool
```

### Use the agent in code
```python
from src.agent.graph import create_graph_with_persistence
import uuid

graph = create_graph_with_persistence()
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

for event in graph.stream(
    {"messages": [("user", "What is the latest research on LLMs?")]},
    config,
    stream_mode="values"
):
    msg = event["messages"][-1]
    if hasattr(msg, "content") and msg.content:
        print(msg.content)
```

### Resume a conversation
```python
# Use the same thread_id from before — LangGraph loads history automatically
config = {"configurable": {"thread_id": "existing-thread-id-here"}}
graph.invoke({"messages": [("user", "Follow-up question")]}, config)
```

### Add a new tool
1. Create `src/tools/my_tool.py` with `@tool` decorator
2. Import and add to `tools` list in both `create_graph()` and `create_graph_with_persistence()` in `graph.py`
3. Add a line to `SYSTEM_PROMPT` describing when to use it

### Delete a conversation thread
```python
from src.persistance.checkpointer import get_checkpointer, clear_thread
checkpointer = get_checkpointer()
clear_thread("thread-id-here", checkpointer)
```

---

## Gotchas & Known Issues

**1. Wrong checkpointer constructor**
```python
# WRONG
SqliteSaver(db_path)

# CORRECT
SqliteSaver.from_conn_string(db_path)
```

**2. Wrong config structure for persistence**
```python
# WRONG - thread never loads
config = {"thread_id": "abc"}

# CORRECT
config = {"configurable": {"thread_id": "abc"}}
```

**3. State definition missing reducer**
```python
# WRONG - new messages replace old ones
messages: list

# CORRECT - new messages merge into history
messages: Annotated[list, add_messages]
```

**4. ArXiv tool type mismatch**
```python
# WRONG - returns List[Document], LLM can't read it
ArxivRetriever()

# CORRECT - returns formatted string
ArxivQueryRun(api_wrapper=ArxivAPIWrapper(...))
```

**5. Known typos in codebase (don't rename without updating all imports)**
- `src/tools/__intit__.py` → should be `__init__.py`
- `src/persistance/` → should be `persistence/`

---

## What Phase 4 (UI) Needs

**`src/ui/streamlit.py`** — build this:
```python
import streamlit as st
import uuid
from src.agent.graph import create_graph_with_persistence

st.title("Research Agent")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

graph = create_graph_with_persistence()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if prompt := st.chat_input("Ask anything"):
    for event in graph.stream({"messages": [("user", prompt)]}, config, stream_mode="values"):
        msg = event["messages"][-1]
        if hasattr(msg, "content") and msg.content:
            st.chat_message("assistant").write(msg.content)
```

**`run.py`** — build this:
```python
import subprocess
subprocess.run(["streamlit", "run", "src/ui/streamlit.py"])
```

Then: `python run.py` to launch.

**Full Phase 4 feature list:**
- Chat messages display (human + assistant)
- Sidebar with thread list (from `list_all_threads()`)
- New Conversation button (generates new UUID)
- Show tool calls as expandable sections
- Streaming token-by-token display

---

## Dependencies

```
langchain                     # Core utilities
langgraph                     # Auto-installed with langchain
langgraph-checkpoint-sqlite   # SQLite checkpointer
langchain-groq                # Groq LLM
langchain-tavily              # Tavily tool wrapper
tavily-python                 # Tavily SDK
wikipedia                     # Wikipedia client
arxiv                         # ArXiv client
streamlit                     # UI (Phase 4)
fastapi + uvicorn             # Future REST API
python-dotenv                 # .env loading
```

---

## Reference Links

- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Groq console: https://console.groq.com
- Tavily API: https://tavily.com
- LangSmith tracing: https://smith.langchain.com
- ReAct paper: https://arxiv.org/abs/2210.03629
- System design doc: `System_design.md` in this repo
- **Framework:** LangGraph (not LangChain - important distinction!)
- **LLM:** Groq (Llama 3.3 70B Versatile)
- **Persistence:** SQLite (via langgraph-checkpoint-sqlite)
- **Tools:** Tavily, ArXiv, Wikipedia, Calculator
- **UI:** Streamlit (Phase 4, not yet implemented)

---

## 🗂️ Project Structure
