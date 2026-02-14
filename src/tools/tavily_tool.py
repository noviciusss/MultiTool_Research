'''
Websearch tool using tavily API
Searches the web for current info 

'''
from langchain_tavily import TavilySearch
import os 
from dotenv import load_dotenv
load_dotenv()

def get_tavily_tool():
    '''
    Create a tavily web seach tool 
    return : TavilySearch:configured search tool
    '''
    return TavilySearch(
        max_results=3,
        search_depth = 'basic', # we have option of advanded serach -->slow but comprehensive
        include_answer = True, #include direct answer in response if available
        include_raw_response = False, #Dont include full html 
        api_keys = os.getenv("TAVILY_API_KEY")
    )
    
if __name__ == "__main__":
    tool = get_tavily_tool()
    result = tool.invoke({"query":"Eu-india trade deal 2026 and its impact on global economy"})
    print("Tavily Search Results:")
    print(result)