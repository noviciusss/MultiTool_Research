# from langchain_community.retrievers import ArxivRetriever
# def get_arxiv_tool():
#     '''
#     Create an arxiv query tool using langchain's ArxivAPIWrapper
#     return : ArxivQueryRun: configured arxiv search tool
#     '''
#     arxiv_wrapper = ArxivRetriever(
#         top_k_results=3,  # Limit to top 3 results for relevance
#         doc_content_chars_max=1000 #limit content length 
#     )
#     return arxiv_wrapper

# if __name__ == "__main__":
#     tool = get_arxiv_tool()
#     result = tool.invoke("quantum computing error correction")
#     print("Arxiv Search Results:")
#     print(result)

#==================Why not use ArxivRetriever directly?====================

# The ArxivRetriever is designed for retrieval tasks and may not fit seamlessly into the tool-based architecture of our agent.
# By wrapping it in an ArxivQueryRun, we can ensure that it behaves like a tool, allowing the agent to invoke it as needed and handle its outputs in a consistent manner with other tools. 
# This abstraction also allows us to easily modify or enhance the arXiv search functionality in the future without affecting the rest of the agent's architecture.

#Return List[Document] but tool aspects str 
#Use retrival in Rag pipelines and best for vector DB and LLm cannot read it directly but with wrapper we can use it as a tool and agent can read the results and use it in its reasoning process.
# ===================== ArXiv Tool =====================


from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


def get_arxiv_tool():
    """
    Create ArXiv paper search tool.
    
    Returns:
        ArxivQueryRun: Configured arXiv search tool
    """
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=3,  # Return top 3 papers
        doc_content_chars_max=1500, # Limit content length
        load_all_available_meta=False
    )
    #Wrap in ArxivQueryRun (formats for llm)
    return ArxivQueryRun(api_wrapper=arxiv_wrapper)


# Test this tool independently
if __name__ == "__main__":
    tool = get_arxiv_tool()
    result = tool.invoke("Quantum Entanglement")
    print("ArXiv Search Results:")
    print(result)
    