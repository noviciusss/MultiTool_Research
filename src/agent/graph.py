from typing import Annotated,TypedDict,Literal

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages 
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

import os
from dotenv import load_dotenv

from src.tools.calculator_tool import calculator
from src.tools.tavily_tool import get_tavily_tool
from src.tools.wikipedia_tool import get_wikipedia_tool
from src.tools.arxiv_tool import get_arxiv_tool

load_dotenv()

# ================================ Define State =============

class AgentState(TypedDict):
    """State of the agent"""
    messages :Annotated[list,add_messages]
    
#============================System Prompt =========================
SYSTEM_PROMPT = """You are a helpful resarch assistant with access to multiple tools.

**Available Tools:**
- tavily_search: Search the web for current information, news, recent events
- arxiv: Search academic papers for scientific research
- wikipedia: Get general knowledge, definitions, historical facts
- calculator: Perform mathematical calculations and statistics

**How to use tools:**
1- think about what information you need
2- choose the RIGHT tool based on the task
3- Use multiple tools if needed
4- Synthesize result into clear answer

**Examples:**
- "Latest quantum computing news" → Use tavily_search
- "Academic papers on quantum error correction" → Use arxiv
- "What is quantum entanglement?" → Use wikipedia
- "Calculate average of [1,2,3,4,5]" → Use calculator
- "Generate Fibonacci sequence" → Use python_executor

Be concise but thorough. Always cite sources when available.

"""
# ==================== Create Nodes =====================

def create_agent_node(tools:list):
    """Factory function to create a agent node with given tools.
    args : tools : list : list of tool instances to provide to agent
    return : function : node function that can be added to graph
    """
    
    llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Fast, high-quality model
            temperature=0,  # Deterministic responses
            api_key=os.getenv("GROQ_API_KEY"),
            
        )
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state:AgentState) ->dict:
        """
        Agent reasoning node.
        
        Decides:
        - Should I call a tool?
        - Which tool(s)?
        - Or should I respond directly?
        """   
        messages = state['messages']
        
        if not any(isinstance(m,SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            
        response = llm_with_tools.invoke(messages)
        return {"messages":[response]}
    return agent_node

#=================Conditional function =================
def should_continue(state:AgentState) -> Literal["tools","end"]:
    """Decide whether to continue with tools or end conversation.
    If agent response contains tool calls, return "tools" to execute them.
    Otherwise, return "end" to finish.
    """
    last_message = state['messages'][-1]
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"
# ==================== Create Graph =====================

def create_graph():
    """Build agent graph with tools and ReAct loop.
    
    Graph flow 
    START
        ↓
    agent_node (decide: use tool or respond?)
        ↓
    should_continue (conditional edge)
        ↓         ↓
    tools       END
        ↓
    agent_node (loop back - agent sees tool results)
        ↓
    should_continue
    """
    tools =[
        get_tavily_tool(),
        get_arxiv_tool(),
        get_wikipedia_tool(),
        calculator
    ]
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node('agent', create_agent_node(tools))
    #ToolNOde automatically executes tools 
    graph_builder.add_node("tools", ToolNode(tools)) ## This node will execute any tool calls in the agent's last message and returns ToolMessage with results  
    
    graph_builder.add_edge(START, 'agent')
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge('tools', 'agent')
    
    return graph_builder.compile()

# ==================== Create Graph with Persistence =====================

def create_graph_with_persistence(db_path: str = "data/checkpoints.db"):
    """Create agent graph with SQLite persistence enabled."""
    from src.persistance.checkpointer import get_checkpointer
    from src.tools.tavily_tool import get_tavily_tool
    from src.tools.arxiv_tool import get_arxiv_tool
    from src.tools.wikipedia_tool import get_wikipedia_tool
    from src.tools.calculator_tool import calculator
    
    tools = [get_tavily_tool(), get_arxiv_tool(), get_wikipedia_tool(), calculator]
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node('agent', create_agent_node(tools))
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, 'agent')
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge('tools', 'agent')
    
    checkpointer = get_checkpointer(db_path)
    return graph_builder.compile(checkpointer=checkpointer)


# ====================Test function ===========

if __name__ == "__main__":
    """
    Test Phase 2 with multiple tools.
    """
    print("🔬 Testing Phase 2: Agent with Tools\n")
    print("=" * 60)
    
    graph = create_graph()
    
    # Test 1: Should use ArXiv
    
    result = graph.invoke({
        "messages": [("user", "What are recent papers on small language models?")]
    })
    
    print("\n💬 Conversation:")
    for msg in result["messages"]:
        if hasattr(msg, 'type'):
            print(f"\n{msg.type.upper()}:")
            if hasattr(msg, 'content') and msg.content:
                print(msg.content[:300] + "..." if len(msg.content) > 300 else msg.content)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"  🔧 Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
    print("\n" + "=" * 60)

    # Test with persistence
    print("\nTesting Agent with Persistence\n" + "="*60)
    
    # Use same thread_id to test resume (change this to test new conversation)
    thread_id = "test_conversation_1"  # Hardcode to test resume
    # thread_id = str(uuid.uuid4())    # Uncomment for new conversation
    
    print(f"Thread ID: {thread_id}\n")
    
    graph = create_graph_with_persistence()
    config = {"configurable": {"thread_id": thread_id}}
    
    query = "What is the square root of 144?"
    print(f"Query: {query}\n")
    
    for event in graph.stream(
        {"messages": [("user", query)]},
        config,
        stream_mode="values"
    ):
        if "messages" in event:
            msg = event["messages"][-1]
            if hasattr(msg, 'type') and msg.type == "ai" and msg.content:
                print(f"Assistant: {msg.content}\n")
    
    print("="*60)
    print(f"✅ Conversation saved! Run again with same thread_id to resume.")
