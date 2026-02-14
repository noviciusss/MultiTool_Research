"""
Wikipedia Tool
search wikipedia for information on a topic
when to use : general knowledge, historical events, Definitions

"""
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def get_wikipedia_tool():
    '''
    Create a wikipedia search tool using langchain's WikipediaAPIWrapper
    return : WikipediaQueryRun: configured wikipedia search tool
    '''
    wikipedia_wrapper = WikipediaAPIWrapper(
        top_k_results=3,  # Limit to top 3 results for relevance
        doc_content_chars_max=1000 #limit content length 
    )
    return WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

if __name__ == "__main__":
    tool = get_wikipedia_tool()
    result = tool.invoke("What is quantum computing?")
    print("Wikipedia Search Results:")
    print(result)