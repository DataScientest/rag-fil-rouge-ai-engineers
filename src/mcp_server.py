"""MCP server (stdio, official `mcp` SDK, FastMCP) that exposes the RAG to any MCP client: search the fiches, or ask a sourced question."""
from mcp.server.fastmcp import FastMCP

from src.generation import answer_question, build_context, check_sources
from src.ingestion import load_sections
from src.retrieval import expand_to_sections, search

mcp = FastMCP("fiches-conges")
SECTIONS = load_sections()


@mcp.tool()
def search_fiches(query: str, k: int = 3) -> str:
    """Search the fiches on employee leave (Service-Public). Returns the k best sections, each preceded by its fiche id."""
    docs = expand_to_sections(search(query, k=k, collection="chunks_titled"), SECTIONS)
    return build_context(docs)


@mcp.tool()
def ask_fiches(question: str) -> str:
    """Answer a question on employee leave from the fiches, with the ids of the fiches used (or NO_ANSWER)."""
    answer, docs = answer_question(question, sections=SECTIONS)
    invented = check_sources(answer, docs)
    text = answer.answer + "\nSources : " + (", ".join(answer.sources) or "aucune")
    if invented:
        text += "\nSources non vérifiées : " + ", ".join(invented)
    return text


if __name__ == "__main__":
    mcp.run(transport="stdio")  # stdout carries the protocol: never print() here
