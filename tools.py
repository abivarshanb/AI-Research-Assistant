
from pathlib import Path

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper


# Initialize search services
duckduckgo = DuckDuckGoSearchRun()

wikipedia = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000
)


@tool
def search_tool(query: str) -> str:
    """Search the internet for information about a topic."""
    return duckduckgo.run(query)


@tool
def wiki_tool(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    return wikipedia.run(query)


@tool
def save_tool(data: str) -> str:
    """Save research information to research_output.txt."""
    file_path = Path(__file__).parent / "research_output.txt"

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(data + "\n\n")

    return f"Research saved to {file_path.name}"
