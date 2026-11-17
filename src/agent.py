"""An agent that decides by itself which tools to call to answer a question."""
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langgraph.errors import GraphRecursionError

from src.generation import Answer, build_context
from src.ingestion import load_sections
from src.retrieval import expand_to_sections, search

load_dotenv()

SECTIONS = load_sections()
MAX_STEPS = 15


@tool(response_format="content_and_artifact")
def search_fiches(query: str) -> tuple[str, list[Document]]:
    """Search the fiches. Give a short query about ONE topic. Returns the matching passages, each preceded by its fiche id."""
    docs = expand_to_sections(search(query, k=3, collection="chunks_titled"), SECTIONS)
    return build_context(docs), docs


@tool(response_format="content_and_artifact")
def read_fiche(doc_id: str) -> tuple[str, list[Document]]:
    """Read a whole fiche from its id, for example F34154."""
    path = "data/corpus/" + doc_id + ".md"
    if not os.path.exists(path):  # the model can pass a title instead of an id: tell it, do not crash
        return "Identifiant de fiche inconnu : " + doc_id + ". Utilisez un identifiant entre crochets, par exemple F2258.", []
    with open(path, encoding="utf-8") as file:
        text = file.read()
    docs = [Document(page_content=text, metadata={"doc_id": doc_id})]
    return build_context(docs), docs


SYSTEM_PROMPT = """Vous répondez aux questions de salariés du secteur privé sur leurs congés, à partir des fiches du corpus.

Outils : search_fiches (chercher des passages) et read_fiche (lire une fiche entière à partir de son identifiant).

Règles :
- Vous devez appeler search_fiches avant de répondre. Si la question touche plusieurs sujets, faites une recherche par sujet.
- Répondez uniquement à partir des passages lus, en français, de façon concise.
- Citez dans "sources" l'identifiant de chaque fiche que vous avez utilisée, par exemple F2381.
- Si les passages ne permettent pas de répondre avec certitude, mettez exactement NO_ANSWER dans "answer" et une liste vide dans "sources"."""


def build_agent():
    """Build the agent: a model, two tools, a system prompt, and an Answer as final output."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"parallel_tool_calls": False},
                     base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    return create_agent(llm, [search_fiches, read_fiche], system_prompt=SYSTEM_PROMPT, response_format=ToolStrategy(Answer))


@traceable(name="ask_agent", process_inputs=lambda inputs: {"question": inputs["question"]})
def ask_agent(question: str, agent=None) -> tuple[Answer, list[Document], list[tuple[str, dict]]]:
    """Ask the agent; return the answer, the passages it read, and the tool calls it made."""
    if agent is None:
        agent = build_agent()
    try:
        state = agent.invoke({"messages": [("user", question)]}, config={"recursion_limit": MAX_STEPS})
    except GraphRecursionError:
        return Answer(answer="NO_ANSWER", sources=[]), [], []
    docs = []
    calls = []
    for message in state["messages"]:
        if message.type == "tool" and message.artifact:
            docs.extend(message.artifact)
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] != "Answer":
                calls.append((call["name"], call["args"]))
    return state["structured_response"], docs, calls


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Combien de jours de congé pour mon mariage ?"
    answer, docs, calls = ask_agent(question)
    for name, args in calls:
        print("Outil :", name, args)
    print(answer.answer)
    print("Sources :", ", ".join(answer.sources) or "aucune")
