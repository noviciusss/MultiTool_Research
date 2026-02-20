# ==================Why not use ArxivRetriever directly?====================
# ArxivRetriever returns List[Document] — LLM can't read that directly.
# Our custom wrapper returns a plain string the LLM understands.
# ArxivQueryRun from langchain_community has a bug in some versions where
# top_k_results isn't forwarded to the API call (sends max_results=100 instead),
# triggering HTTP 429 rate limits. We bypass this by calling arxiv directly.
# =========================================================================

import arxiv
import time
from langchain_core.tools import tool


@tool
def arxiv_search(query: str) -> str:
    """
    Search ArXiv for academic papers on a topic.
    Use for: scientific research, academic papers, technical topics.

    Args:
        query: Search query string (e.g. "small language models 2024")

    Returns:
        Formatted string with paper titles, authors, and summaries.
    """
    try:
        client = arxiv.Client(
            page_size=2,        # Only fetch 2 from API — prevents 429
            delay_seconds=3,    # Polite delay between retries
            num_retries=1,      # Don't hammer the API on failure
        )

        search = arxiv.Search(
            query=query,
            max_results=2,      # Explicitly cap at 2
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            summary = paper.summary.replace("\n", " ")
            if len(summary) > 600:
                summary = summary[:600] + "..."

            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {summary}\n"
                f"URL: {paper.entry_id}\n"
            )

        if not results:
            return "No papers found for this query. Try different keywords."

        return "\n---\n".join(results)

    except arxiv.HTTPError as e:
        if "429" in str(e):
            return (
                "ArXiv is rate limiting requests right now (too many requests). "
                "Try again in 30 seconds, or use a more specific query."
            )
        return f"ArXiv API error: {str(e)}"

    except Exception as e:
        return f"Could not fetch ArXiv results: {str(e)}"


def get_arxiv_tool():
    """Return the arxiv_search tool for use in the agent."""
    return arxiv_search


if __name__ == "__main__":
    result = arxiv_search.invoke("small language models efficiency")
    print(result)
