# 🏗️ System Design for Project 1: Multi-Tool Research Agent

## 1️⃣ How to Think About Architecture (Mental Model)

### **The Core Question: "What is the journey of a single user message?"**

When designing any system, trace ONE request from start to finish:

```
User types question → ??? → User sees answer
```

Fill in the `???` by asking these questions in order:

#### **Layer 1: User Interface (The Front Door)**
- *Where does the user interact?* → Web UI (Streamlit), CLI, API
- *What do they send?* → Text message
- *What do they expect back?* → Streaming answer with agent thoughts

#### **Layer 2: Business Logic (The Brain)**
- *Who processes the request?* → LangGraph agent
- *What decisions are made?* → Which tools to call, when to stop
- *What's the state?* → Conversation history, tool results, current step

#### **Layer 3: External Services (The Hands)**
- *What external help is needed?* → LLM API (Groq), search APIs (Tavily), databases
- *How do we call them?* → Tools (LangChain tool wrappers)
- *What do they return?* → Raw data that agents interpret

#### **Layer 4: Persistence (The Memory)**
- *What needs to survive crashes?* → Conversation history, user sessions
- *Where is it stored?* → PostgreSQL/SQLite (checkpointer)
- *How is it retrieved?* → Thread ID (conversation identifier)

#### **Layer 5: Observability (The Eyes)**
- *How do we debug?* → LangSmith traces, logs
- *What metrics matter?* → Latency, cost, tool success rate
- *When things break?* → Error logs, stack traces

---

### **The "Onion Model" of System Design**

Think of your system as layers around the core:

```
┌─────────────────────────────────────┐
│  User Interface (Streamlit/FastAPI) │ ← What user sees
├─────────────────────────────────────┤
│  Application Layer (graph.run())    │ ← Orchestration logic
├─────────────────────────────────────┤
│  Agent Core (LangGraph StateGraph)  │ ← Decision-making
├─────────────────────────────────────┤
│  Tools Layer (Search, Wikipedia)    │ ← Actions agents take
├─────────────────────────────────────┤
│  LLM Layer (Groq, OpenAI APIs)      │ ← Reasoning engine
├─────────────────────────────────────┤
│  Persistence (PostgreSQL)           │ ← Long-term memory
└─────────────────────────────────────┘
```

**Key Principle:** Each layer only talks to adjacent layers. UI doesn't directly call LLM; it goes through the agent.

---

### **Why This Matters**

**Beginner mistake:** Writing everything in one file, mixing UI code with agent logic with database calls.

**Pro approach:** Separate concerns so you can:
- Swap Streamlit for FastAPI without touching agent code
- Change from Groq to Claude without rewriting tools
- Test agent logic without running the UI
- Scale each layer independently

---

## 2️⃣ System Design for Project 1

### **High-Level Architecture**

```
┌──────────────────────────────────────────────────────────────┐
│                         USER                                  │
│                           ↓                                   │
│                  [Streamlit Web UI]                           │
│                  - Input text                                 │
│                  - Display streaming responses                │
│                  - Show agent thoughts                        │
│                           ↓                                   │
│                  [Session Manager]                            │
│                  - Thread ID per user                         │
│                  - Loads conversation history                 │
│                           ↓                                   │
│              ┌────────────────────────┐                       │
│              │  LANGGRAPH AGENT CORE  │                       │
│              │  (ReAct Pattern)       │                       │
│              └────────────────────────┘                       │
│                     ↓         ↓                               │
│            [Agent Node]  [Tool Node]                          │
│                 ↓              ↓                               │
│         [LLM API (Groq)]  [LangChain Tools]                   │
│                              ↓    ↓    ↓                      │
│                         [Tavily] [ArXiv] [Wikipedia]          │
│                                                                │
│                  [PostgreSQL/SQLite]                           │
│                  - Checkpoints (conversation history)          │
│                  - State snapshots                            │
│                                                                │
│                  [LangSmith] (Optional)                        │
│                  - Trace every agent decision                 │
│                  - Debug tool calls                           │
└──────────────────────────────────────────────────────────────┘
```

---

### **Detailed Component Breakdown**

#### **A. Frontend Layer (Streamlit)**

```python
# What it does:
# 1. Collects user input
# 2. Manages thread_id (session identifier)
# 3. Streams agent outputs to UI
# 4. Displays agent thoughts (tool calls, reasoning)

Components:
├── st.chat_input() → Get user message
├── st.chat_message() → Display conversation
├── Streaming handler → Show agent steps in real-time
└── Session state → Store thread_id
```

**Why Streamlit first?**
- Fastest to build (no HTML/CSS)
- Built-in session management
- Easy streaming display
- Later: Can replace with FastAPI + Next.js

---

#### **B. Agent Core (LangGraph)**

This is where the magic happens. The **ReAct pattern**:

```python
# STATE DEFINITION
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    # LangGraph auto-manages this list

# GRAPH STRUCTURE
StateGraph flow:

START
  ↓
[agent_node]  ← LLM decides: call tool OR answer directly
  ↓
{conditional_edge}
  ↓     ↓
[tools] [END]
  ↓
[agent_node]  ← Loop: Agent sees tool results, decides next step
  ↓
  ...
```

**The ReAct Loop (Reason + Act):**

1. **Reason:** LLM sees conversation + available tools → decides action
2. **Act:** If tool needed → `ToolNode` executes it
3. **Observe:** Tool results added to state
4. **Repeat:** LLM sees results → decides next action
5. **Finish:** When LLM satisfied → responds to user

---

#### **C. Tool Layer**

```python
Tools are functions the agent can call:

[Tavily Search Tool]
  - Input: "quantum computing breakthroughs 2025"
  - Output: List of web results with snippets

[ArXiv Tool]
  - Input: "quantum computing 2025"
  - Output: Recent papers with abstracts

[Wikipedia Tool]
  - Input: "Quantum computing"
  - Output: Summary article

[Calculator Tool]
  - Input: "sqrt(144)"
  - Output: "12"
```

**How LangGraph binds tools:**

```python
# Agent gets tool descriptions
tools = [TavilySearch(), WikipediaQuery(), ArxivSearch()]

# LLM sees tool schemas (function name, params, description)
llm_with_tools = llm.bind_tools(tools)

# When LLM wants to use a tool, it returns:
{
  "tool": "tavily_search",
  "arguments": {"query": "quantum computing 2025"}
}

# ToolNode automatically executes and returns results
```

---

#### **D. Persistence Layer (PostgreSQL/SQLite)**

**What gets saved:**

```python
Every time state changes, LangGraph saves:

Checkpoint = {
  "thread_id": "user_123_session_456",
  "checkpoint_id": "uuid-xxx",
  "state": {
    "messages": [
      {"role": "user", "content": "What is quantum computing?"},
      {"role": "assistant", "content": "Let me search..."},
      {"role": "tool", "tool": "wikipedia", "result": "..."},
      {"role": "assistant", "content": "Quantum computing is..."}
    ]
  },
  "metadata": {"step": 3, "timestamp": "2026-02-14T10:30:00"}
}
```

**Why checkpointing matters:**
- User refreshes page → Conversation reloads from database
- Agent crashes mid-execution → Resume from last checkpoint
- Debug issues → Replay conversation step-by-step

---

#### **E. Observability (LangSmith)**

**What it captures:**

```
LangSmith Trace:

Run 1: User asks question
├── Run 2: Agent node (LLM call)
│   └── LLM decides to call tavily_search
├── Run 3: Tool node executes tavily_search
│   └── Returns 5 web results
├── Run 4: Agent node (LLM call)
│   └── LLM decides to call arxiv_search
├── Run 5: Tool node executes arxiv_search
│   └── Returns 3 papers
└── Run 6: Agent node (LLM call)
    └── LLM synthesizes answer → END

Total latency: 8.3s
Total cost: $0.012
Tool calls: 2
```

**Why this is critical:**
- See exactly why agent made each decision
- Identify slow tools (optimize them)
- Track costs per conversation
- Debug unexpected behavior

---

## 3️⃣ How It All Works Together (Complete Data Flow)

Let me trace a **real user query** through the entire system step-by-step.

### **Example: User asks "What are the latest quantum computing breakthroughs in 2025?"**

---

#### **Step 1: User Input (Streamlit)**

```python
# streamlit_app.py
if user_input := st.chat_input("Ask me anything"):
    # Generate or retrieve thread_id
    thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))
    
    # Display user message
    st.chat_message("user").write(user_input)
    
    # Call agent with streaming
    config = {"configurable": {"thread_id": thread_id}}
    
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        for event in graph.stream(
            {"messages": [("user", user_input)]},
            config=config,
            stream_mode="values"
        ):
            # Update UI with each agent step
            stream_placeholder.markdown(format_messages(event))
```

**What happens:**
- User types question → Streamlit captures it
- Thread ID created/retrieved (identifies this conversation)
- Question sent to LangGraph agent
- UI prepared to stream responses

---

#### **Step 2: Agent Reasoning (LangGraph - First Call)**

```python
# agent.py - agent_node function
def agent_node(state: AgentState):
    messages = state["messages"]
    
    # LLM sees:
    # - Conversation history
    # - Available tools (Tavily, ArXiv, Wikipedia)
    # - System prompt explaining its role
    
    response = llm_with_tools.invoke(messages)
    
    # LLM returns:
    # "I should search for recent quantum computing news"
    # Tool call: tavily_search("quantum computing breakthroughs 2025")
    
    return {"messages": [response]}
```

**LLM's internal reasoning (simplified):**
```
User wants latest breakthroughs in 2025.
I need current information (beyond my training data).
Tools available: Tavily (web search), ArXiv (papers), Wikipedia (general)
Best choice: Tavily for recent news
Action: Call tavily_search with query "quantum computing breakthroughs 2025"
```

---

#### **Step 3: Tool Execution (ToolNode)**

```python
# LangGraph's built-in ToolNode
def tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract tool calls from LLM response
    tool_calls = last_message.tool_calls
    # tool_calls = [{"name": "tavily_search", "args": {...}}]
    
    # Execute each tool
    tool_outputs = []
    for tool_call in tool_calls:
        tool = tools[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        tool_outputs.append(result)
    
    # Return results as ToolMessage
    return {"messages": tool_outputs}
```

**Tavily API returns:**
```json
[
  {
    "title": "IBM Unveils 1000-Qubit Quantum Processor",
    "url": "https://...",
    "content": "IBM announced breakthrough in error correction..."
  },
  {
    "title": "Google Achieves Quantum Supremacy in 2025",
    "url": "https://...",
    "content": "Google's new quantum chip solves..."
  },
  ...
]
```

---

#### **Step 4: State Update (Checkpointer Saves)**

```python
# After tool execution, state looks like:
state = {
  "messages": [
    HumanMessage("What are the latest quantum computing breakthroughs in 2025?"),
    AIMessage("I'll search for recent information", tool_calls=[...]),
    ToolMessage(name="tavily_search", content="IBM Unveils 1000-Qubit... Google Achieves...")
  ]
}

# PostgresSaver automatically saves this checkpoint
# Database now has:
INSERT INTO checkpoints (thread_id, checkpoint_id, state, ...)
VALUES ('user_123', 'checkpoint_xyz', {...}, ...)
```

**Why save here?** If agent crashes after tool call, we don't re-execute expensive search.

---

#### **Step 5: Agent Reasoning (Second Call)**

```python
# agent_node called again with updated state
def agent_node(state: AgentState):
    messages = state["messages"]  # Now includes tool results
    
    response = llm_with_tools.invoke(messages)
    
    # LLM sees tool results and thinks:
    # "I have web results about IBM and Google breakthroughs"
    # "Let me also check academic papers for more depth"
    # Action: Call arxiv_search
    
    return {"messages": [response]}
```

---

#### **Step 6: Second Tool Call (ArXiv)**

```python
# ToolNode executes arxiv_search
arxiv_tool.invoke({"query": "quantum computing 2025"})

# Returns:
[
  {
    "title": "Advances in Quantum Error Correction",
    "authors": "Smith et al.",
    "abstract": "We present new techniques for...",
    "published": "2025-01-15"
  },
  ...
]
```

---

#### **Step 7: Agent Final Decision**

```python
# agent_node called third time
def agent_node(state: AgentState):
    messages = state["messages"]
    # Now has: user query + tavily results + arxiv results
    
    response = llm_with_tools.invoke(messages)
    
    # LLM thinks:
    # "I have comprehensive information from web and papers"
    # "Time to synthesize final answer"
    # Action: NO TOOL CALL (just respond)
    
    return {"messages": [response]}
```

**Conditional edge logic:**
```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # More tools to execute
    else:
        return "END"  # Agent finished, return to user
```

Agent has no tool calls → Graph routes to END

---

#### **Step 8: Stream Response to User**

```python
# Streamlit receives final state
final_state = {
  "messages": [
    ...,
    AIMessage("Based on recent developments, here are the key quantum computing breakthroughs in 2025:\n\n1. **IBM's 1000-Qubit Processor**...\n2. **Google's Quantum Supremacy**...\n3. **Error Correction Advances** (from ArXiv papers)...")
  ]
}

# Streamlit displays formatted response
st.chat_message("assistant").markdown(final_state["messages"][-1].content)
```

---

#### **Step 9: Persistence Saves Final State**

```python
# PostgreSQL checkpoint updated:
{
  "thread_id": "user_123",
  "checkpoint_id": "final_xyz",
  "state": {
    "messages": [/* full conversation */]
  }
}

# User can refresh page tomorrow → conversation still there
```

---

### **Visual Flow Diagram**

```
USER: "What are the latest quantum computing breakthroughs in 2025?"
  ↓
[Streamlit UI]
  ↓
(thread_id="user_123", message)
  
[Graph.stream()]
  ↓
┌─────────────────────┐
│   Agent Node (1)    │ ← LLM: "I need current info, call Tavily"
└─────────────────────┘
  ↓
(tool_call: tavily_search)
  
┌─────────────────────┐
│    Tool Node        │ → Tavily API → Returns web results
└─────────────────────┘
  ↓
(ToolMessage with results)

[Checkpointer saves state]
  ↓
┌─────────────────────┐
│   Agent Node (2)    │ ← LLM: "Good start, let me check papers too"
└─────────────────────┘
  ↓
(tool_call: arxiv_search)

┌─────────────────────┐
│    Tool Node        │ → ArXiv API → Returns papers
└─────────────────────┘
  ↓
(ToolMessage with papers)

[Checkpointer saves state]
  ↓
┌─────────────────────┐
│   Agent Node (3)    │ ← LLM: "I have enough info, synthesize answer"
└─────────────────────┘
  ↓
(no tool calls → END)

[Conditional: should_continue() returns "END"]
  ↓
[Streamlit displays final answer]
  ↓
USER sees: "Based on recent developments, here are the key breakthroughs..."
```

---

## 🎯 Key Takeaways for System Design Thinking

### **1. Separation of Concerns**
- **UI** only handles display, not logic
- **Agent** only makes decisions, doesn't know about UI
- **Tools** are independent, swappable
- **Persistence** is abstracted (swap SQLite for PostgreSQL without code changes)

### **2. State is Central**
Everything flows through `AgentState`:
- Messages accumulate (conversation history)
- Each node transforms state
- Checkpointer saves state snapshots
- State is the "single source of truth"

### **3. Graph Controls Flow**
```python
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {...})
graph.add_edge("tools", "agent")  # Loop back after tools
```
This structure makes behavior **predictable and debuggable**.

### **4. Streaming is Built-In**
```python
for event in graph.stream(...):
    # Event = state after each node execution
    # Update UI incrementally
```
No manual threading or async complexity needed.

### **5. Observability from Day 1**
- Checkpoints = time-travel debugging
- LangSmith = see every decision
- Logs = catch errors

Don't add these later—build with them.

---

## 🚀 Your Starting Point

**Start here:**

1. **Draw your graph on paper:**
   - What nodes? (agent, tools)
   - What edges? (agent→tools, tools→agent, agent→END)
   - What's in the state? (messages)

2. **Write state definition:**
   ```python
   class AgentState(TypedDict):
       messages: Annotated[list, add_messages]
   ```

3. **Build simplest working version:**
   - 1 agent node (just OpenAI/Groq chat)
   - 1 tool (Wikipedia)
   - Basic Streamlit UI
   - In-memory checkpointer

4. **Test it works end-to-end**

5. **Add complexity incrementally:**
   - Add more tools
   - Switch to PostgreSQL
   - Add LangSmith
   - Improve UI

---

## 📁 Recommended File Structure

```
research_agent/
├── agent.py              # LangGraph agent definition
│   ├── AgentState (TypedDict)
│   ├── agent_node()
│   ├── should_continue()
│   └── create_graph()
│
├── tools.py              # Custom tool implementations
│   ├── TavilySearchTool
│   ├── ArxivSearchTool
│   └── WikipediaTool
│
├── app.py                # Streamlit/FastAPI frontend
│   ├── UI rendering
│   ├── Session management
│   └── Streaming display
│
├── config.py             # Configuration
│   ├── API keys
│   ├── Model settings
│   └── Tool configs
│
├── Dockerfile            # Containerization
├── docker-compose.yml    # PostgreSQL + app
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
└── README.md             # Documentation
```

---

## 🔍 Detailed File Responsibilities

### **agent.py** (The Brain)
- Defines state schema
- Implements agent reasoning logic
- Builds the LangGraph
- NO UI code
- NO database connection code
- Pure business logic

### **tools.py** (The Hands)
- Each tool is a separate class/function
- Implements tool-specific logic
- Error handling per tool
- Can be tested independently

### **app.py** (The Face)
- Streamlit UI components
- Manages user sessions
- Calls `graph.stream()`
- Displays results
- NO agent logic here

### **config.py** (The Settings)
- Centralizes all configuration
- Environment variable loading
- Model selection
- Tool API keys

---

## ⚙️ Technology Stack Decisions

| **Component** | **Choice** | **Why** |
|---------------|------------|---------|
| **LLM** | Groq (Llama 3.1) | Fast inference, free tier |
| **Framework** | LangGraph | Native multi-agent support |
| **UI** | Streamlit | Fastest prototyping |
| **Database** | SQLite → PostgreSQL | Start simple, scale later |
| **Monitoring** | LangSmith | Official LangChain tool |
| **Deployment** | Railway/Render | Easy first deployment |
| **Containerization** | Docker | Industry standard |

---

## 🎓 Learning Path for Implementation

### **Phase 1: Core Agent (Days 1-2)**
- [ ] Setup project structure
- [ ] Create basic agent with 1 tool
- [ ] Test locally with in-memory state
- [ ] Verify ReAct loop works

### **Phase 2: Multiple Tools (Days 3-4)**
- [ ] Add Tavily, ArXiv, Wikipedia
- [ ] Test tool selection logic
- [ ] Handle tool errors gracefully
- [ ] Add streaming to terminal

### **Phase 3: Persistence (Day 5)**
- [ ] Setup SQLite checkpointer
- [ ] Test conversation resume
- [ ] Add thread management
- [ ] Verify state saves correctly

### **Phase 4: UI (Days 6-7)**
- [ ] Build Streamlit interface
- [ ] Implement streaming display
- [ ] Add session management
- [ ] Polish UX (loading states, errors)

### **Phase 5: Production Prep (Days 8-10)**
- [ ] Switch to PostgreSQL
- [ ] Add LangSmith tracing
- [ ] Dockerize application
- [ ] Deploy to Railway/Render

### **Phase 6: Polish (Days 11-14)**
- [ ] Add error handling
- [ ] Improve prompts
- [ ] Write documentation
- [ ] Create demo video

---

## 🚨 Common Pitfalls to Avoid

### **1. Mixing Layers**
❌ Bad: Streamlit code calling LLM directly
✅ Good: Streamlit → graph.stream() → agent handles LLM

### **2. No State Management**
❌ Bad: Storing messages in Python list
✅ Good: Using LangGraph state + checkpointer

### **3. Ignoring Errors**
❌ Bad: Agent crashes on tool failure
✅ Good: Try/except in tools, fallback strategies

### **4. No Observability**
❌ Bad: Can't debug why agent made decision
✅ Good: LangSmith traces from day 1

### **5. Hardcoded Values**
❌ Bad: API keys in code
✅ Good: Environment variables + config.py

---

## 📚 Essential Resources

**Official Documentation:**
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph How-To Guides](https://langchain-ai.github.io/langgraph/how-tos/)
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools/)

**Key Examples to Study:**
- [ReAct Agent Example](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Persistence Tutorial](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [Streaming How-To](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)

**Deployment:**
- [Railway Deployment Guide](https://docs.railway.app/)
- [LangSmith Setup](https://docs.smith.langchain.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 🎯 Success Criteria

You'll know you've mastered this when you can:

- ✅ Explain the data flow from user input to final response
- ✅ Add a new tool in under 10 minutes
- ✅ Debug agent decisions using LangSmith
- ✅ Swap UI from Streamlit to FastAPI
- ✅ Switch LLM providers without breaking code
- ✅ Deploy to production with confidence
- ✅ Explain ReAct pattern to someone else
- ✅ Handle edge cases (tool failures, API limits)

---

## 🤔 Questions to Ask Yourself During Build

1. **"Can I test this component in isolation?"**
   - If not, decouple it

2. **"What happens if this API call fails?"**
   - Add error handling

3. **"How will I debug this in production?"**
   - Add logging/tracing

4. **"Can I swap this dependency easily?"**
   - If not, add abstraction layer

5. **"Will this scale to 100 users?"**
   - If not, identify bottlenecks early

---

## 🔄 Iterative Development Strategy

Don't build everything at once. Follow this sequence:

```
Version 0.1: Simplest possible agent
  → 1 node, 1 tool, terminal output

Version 0.2: Add tool variety
  → 3+ tools, better prompting

Version 0.3: Add persistence
  → SQLite checkpointer

Version 0.4: Add UI
  → Basic Streamlit interface

Version 0.5: Production-ize
  → PostgreSQL, Docker, deployment

Version 1.0: Polish
  → Error handling, monitoring, docs
```

Each version should **work end-to-end** before moving forward.

---

## 💡 Next Steps

Ready to implement? I can help you:

1. **Setup the project structure** with actual code scaffolding
2. **Write the core agent logic** with detailed explanations
3. **Implement specific tools** you want to include
4. **Build the Streamlit UI** with streaming
5. **Configure deployment** for Railway/Render

Which would you like to start with?
