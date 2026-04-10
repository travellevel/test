from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from datetime import datetime

# ----- 搜索工具 -----
search_run = DuckDuckGoSearchRun()
@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    return search_run.run(query)

# ----- 维基百科工具 -----
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_run = WikipediaQueryRun(api_wrapper=api_wrapper)
@tool
def wiki_tool(query: str) -> str:
    """Query Wikipedia for information."""
    return wiki_run.run(query)

# ----- 保存工具 -----
@tool
def save_tool(data: str, filename: str = "research_output.txt") -> str:
    """
    Saves structured research data to a text file.
    Args:
        data: The text data to save.
        filename: The name of the file to save to. Defaults to 'research_output.txt'.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"