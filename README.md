# Multi-Tool Research Agent

> A conversational AI research assistant that reasons over multiple data sources using the ReAct pattern — search the web, find papers, look up facts, and do math, all in one loop with persistent memory across sessions.

---

## Overview

```
User Question
      │
      ▼
 ┌─────────────────────────────┐
 │         Agent (LLM)         │  ← Groq / Llama 3.3 70B
 │   "Which tool do I need?"   │
 └────────────┬────────────────┘
              │
     ┌────────▼────────┐
     │  Tool Selection │  ← LLM decides, not hard-coded rules
     └────────┬────────┘
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼         ▼
 Tavily    ArXiv    Wikipedia  Calculator
 (web)    (papers)  (facts)    (math)
    │         │          │         │
    └─────────┴──────────┴─────────┘
              │
              ▼
       Results returned
              │
              ▼
 ┌─────────────────────────────┐
 │         Agent (LLM)         │  ← "Do I need more info?"
 │     Synthesizes answer      │
 └────────────┬────────────────┘
              │  (loops until satisfied)
              ▼
      Final Answer to User
              │
              ▼
       SQLite Checkpoint        ← Saved for future sessions
```

---

## Architecture

### The ReAct Pattern

ReAct stands for **Reason + Act**. Instead of answering in a single pass, the agent iterates:

1. **Reason** — LLM sees the question and available tools, decides what to do
2. **Act** — Calls a tool, gets results
3. **Observe** — LLM sees the result, decides if more info is needed
4. **Repeat** — Until the LLM is satisfied
5. **Respond** — Final synthesized answer

This is a cyclic graph — which is why LangGraph is used instead of LangChain (LangChain is linear).

### State Machine

```
START
  │
  ▼
[agent node] ──── has tool_calls? ──── YES ──► [tool node]
     ▲                                              │
     │                                              │
     └──────────────────────────────────────────────┘
     │
     └──── no tool_calls? ──► END
```

### Layer Model

```
┌─────────────────────────────────────────┐
│            Streamlit UI                 │  src/ui/streamlit.py
├─────────────────────────────────────────┤
│         LangGraph Agent Core            │  src/agent/graph.py
├─────────────────────────────────────────┤
│              Tool Layer                 │  src/tools/*.py
├─────────────────────────────────────────┤
│        Groq LLM (Llama 3.3 70B)        │  via langchain-groq
├─────────────────────────────────────────┤
│         SQLite Persistence              │  src/persistance/checkpointer.py
└─────────────────────────────────────────┘
```

Each layer only talks to adjacent layers. The UI does not call tools directly. The agent does not know about the UI.

---

## Tech Stack

| Component | Choice | Purpose |
|-----------|--------|---------|
| Agent framework | LangGraph | Cyclic graph support, built-in checkpointing |
| LLM | Groq — Llama 3.3 70B Versatile | Fast inference, free tier |
| Web search | Tavily | LLM-optimized results, structured output |
| Paper search | arxiv (direct) | Academic database, no API key needed |
| General knowledge | Wikipedia | Definitions, historical facts |
| Math | Custom `@tool` with restricted `eval()` | Safe expression evaluator |
| Persistence | SQLite via `langgraph-checkpoint-sqlite` | Zero setup, file-based |
| UI | Streamlit | Fastest path to working chat interface |
| Config | python-dotenv | Standard `.env` loading |

---

## Project Structure

```
MultiTool_Research/
├── .env                        # API keys — never commit
├── run.py                      # Entry point: python run.py
├── requirements.txt
├── System_design.md            # Phase-by-phase design rationale
├── claude.md                   # AI assistant context for future sessions
│
├── data/
│   └── checkpoints.db          # SQLite — auto-created on first run
│
└── src/
    ├── agent/
    │   └── graph.py            # Core: AgentState, ReAct graph, persistence
    ├── tools/
    │   ├── tavily_tool.py
    │   ├── arxiv_tool.py
    │   ├── wikipedia_tool.py
    │   └── calculator_tool.py
    ├── persistance/
    │   └── checkpointer.py     # SQLite checkpoint management
    └── ui/
        └── streamlit.py        # Chat interface
```

---

## Setup

**1. Clone and create environment**
```bash
git clone https://github.com/noviciusss/MultiTool_Research.git
cd MultiTool_Research
conda create -n agent_env python=3.11
conda activate agent_env
pip install -r requirements.txt
```

**2. Configure API keys**
```bash
cp .env.example .env
# Edit .env and add:
# GROQ_API_KEY=your_key       → console.groq.com (free)
# TAVILY_API_KEY=your_key     → tavily.com (free tier)
```

**3. Run**
```bash
python run.py
# Opens browser at http://localhost:8501
```

ArXiv and Wikipedia require no API keys.

---

## Usage

Once the UI loads:

- Type any research question in the chat input
- The agent selects tools automatically based on context
- Tool calls are shown in expandable sections
- Conversations persist across browser refreshes
- Use the sidebar to start new conversations or switch between past ones

**Query examples by tool:**

| Query type | Tool used |
|---|---|
| "Latest news on..." | Tavily |
| "Academic papers on..." | ArXiv |
| "What is / Who is / History of..." | Wikipedia |
| "Calculate / What is 15% of..." | Calculator |
| Complex queries | Multiple tools in sequence |

---

## Design Decisions

<details>
<summary><strong>Tradeoffs</strong></summary>

### LangGraph vs LangChain

LangGraph was chosen over LangChain for the agent framework.

LangChain chains are linear — input flows through a sequence and exits. The ReAct pattern requires a loop: agent calls tool, sees results, decides to call another tool or stop. Implementing this in LangChain requires hacky workarounds (AgentExecutor) that are hard to debug and don't support native checkpointing.

LangGraph is a state machine where nodes can loop. The graph explicitly encodes:
```
tools → agent → (loop or end)
```
This makes the control flow transparent and checkpointing automatic.

**Cost:** Steeper learning curve than LangChain.

---

### LLM-based tool selection vs rule-based routing

The LLM decides which tool to call based on the query. The alternative is hard-coded rules:
```python
if "latest" in query: use_tavily()
elif "paper" in query: use_arxiv()
```

Rule-based routing is fast and predictable but breaks on edge cases. "Latest research papers" should use ArXiv, not Tavily — a rule matching "latest" would get it wrong. The LLM handles nuance correctly because it understands context.

**Cost:** Every decision requires an LLM call. At Groq's speeds (500+ tokens/sec) this adds ~1 second per round, which is acceptable.

---

### Groq vs OpenAI

Groq's hosted Llama 3.3 70B is ~10x faster than OpenAI's GPT-4o and free up to generous limits. Quality is comparable for tool selection and synthesis tasks.

OpenAI has better reasoning on complex multi-step problems but at $0.15-0.30 per request the cost is prohibitive for a development project. Groq can be swapped for OpenAI by changing two lines in `graph.py`.

---

### SQLite vs PostgreSQL

SQLite is file-based and requires zero infrastructure. For a single-user agent, it handles all workloads easily.

PostgreSQL would be necessary for concurrent multi-user access. The interface is identical — swapping requires changing one line in `checkpointer.py`:
```python
# Current
SqliteSaver(sqlite3.connect(db_path))

# Future
PostgresSaver.from_conn_string(os.getenv("DATABASE_URL"))
```

**Cost:** SQLite cannot handle concurrent writes from multiple processes.

---

### Direct `arxiv` library vs `langchain_community.ArxivQueryRun`

`ArxivQueryRun` from langchain_community has a bug in current versions: the `top_k_results` parameter is not forwarded to the underlying API call. The library sends `max_results=100` regardless, which triggers HTTP 429 rate limiting from ArXiv.

The fix is to use the `arxiv` library directly with explicit `page_size=2` and `max_results=2` on the `arxiv.Client` and `arxiv.Search` objects respectively. Error handling for 429 and 503 is added so a rate-limit returns a readable message to the LLM instead of crashing the agent.

---

### `SqliteSaver(conn)` vs `SqliteSaver.from_conn_string(path)`

In `langgraph-checkpoint-sqlite >= 2.0`, `from_conn_string()` returns a `_GeneratorContextManager` object — it is designed to be used as a context manager (`with` block), not assigned directly. Passing it to `graph.compile(checkpointer=...)` raises `TypeError: Invalid checkpointer`.

The fix is to create the connection manually:
```python
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

`check_same_thread=False` is required because Streamlit runs on multiple threads and SQLite connections are thread-locked by default.

---

### `st.cache_resource` for graph and checkpointer

Streamlit reruns the entire script on every user interaction — every keypress, every button click. Without `@st.cache_resource`, the graph would be rebuilt and the database connection reopened on every interaction. This would take 10-15 seconds per interaction.

`@st.cache_resource` stores the return value in process memory and returns the cached object on subsequent calls. The graph and checkpointer are initialized once at startup and reused for the lifetime of the server process.

**Important:** Changing tool files requires a full server restart (`Ctrl+C` → `python run.py`) because the cache holds the old compiled graph in memory. Streamlit's file-change hot-reload triggers a script rerun but `cache_resource` returns the cached object without rebuilding.

</details>

<details>
<summary><strong>Findings & Implementation Notes</strong></summary>

### `add_messages` reducer is mandatory

```python
# Wrong — new messages replace entire history
class AgentState(TypedDict):
    messages: list

# Correct — new messages are appended to existing history
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

Without the reducer, every agent turn overwrites the previous conversation. The agent loses all context after the first message.

---

### Config structure for persistence is strict

```python
# Wrong — thread never loads, fresh state every time
config = {"thread_id": "abc"}

# Correct
config = {"configurable": {"thread_id": "abc"}}
```

LangGraph requires all runtime configuration inside a `"configurable"` key. Passing `thread_id` at the top level is silently ignored — no error, just no persistence.

---

### `stream_mode="values"` vs `"updates"`

`stream_mode="values"` emits the **full state** after each node completes. This means:
- After the agent node: you get the full message list including the new tool call decision
- After the tool node: you get the full message list including tool results

`stream_mode="updates"` emits only the **delta** — what changed in that node. More efficient but requires manual merging.

`"values"` was chosen because it is simpler to work with in Streamlit. The last message in the list is always the most recent event.

---

### Tool errors must return strings, not raise exceptions

LangGraph's `ToolNode` has a default error handler that re-raises exceptions. If a tool raises an unhandled exception, it propagates to Streamlit and crashes the UI.

Tools should catch all exceptions and return error strings:
```python
except arxiv.HTTPError as e:
    return "ArXiv is rate limiting. Try again in 30 seconds."
```

The LLM receives this string as the tool result and can relay it to the user gracefully.

---

### ArXiv imposes rate limits regardless of `max_results`

The ArXiv public API enforces HTTP 429 (Too Many Requests) and occasionally 503 (Service Unavailable) when queries come too fast or request too many results. The `arxiv` Python library retries automatically — but each retry compounds the problem.

Setting `num_retries=1` on `arxiv.Client` ensures at most two attempts before failing. The error is then caught and returned as a string.

---

### Streamlit `st.rerun()` outside of a conditional block causes infinite loops

Calling `st.rerun()` unconditionally at script level causes the script to restart immediately every time it runs — an infinite loop that renders a blank page.

`st.rerun()` must only be called inside a user-triggered condition:
```python
# Wrong — infinite loop
with st.sidebar:
    st.title("Agent")
    st.rerun()  # Called on every render

# Correct — only called when button is clicked
if st.button("New Conversation"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()
```

</details>

---

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core agent — LLM + ReAct graph | Complete |
| 2 | Multi-tool integration — Tavily, ArXiv, Wikipedia, Calculator | Complete |
| 3 | Persistence — SQLite checkpointing, thread management | Complete |
| 4 | Streamlit UI — chat interface, sidebar, streaming | Complete |
| 5 | Deployment — containerization, PostgreSQL, auth | Planned |

---

## Environment Variables

| Variable | Required | Source |
|---|---|---|
| `GROQ_API_KEY` | Yes | console.groq.com |
| `TAVILY_API_KEY` | Yes | tavily.com |
| `LANGSMITH_API_KEY` | No | smith.langchain.com |
| `LANGSMITH_TRACING_V2` | No | `true` to enable tracing |

---

## References

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Groq API](https://console.groq.com/docs)
- [Tavily Search API](https://docs.tavily.com)
- [Streamlit documentation](https://docs.streamlit.io)
