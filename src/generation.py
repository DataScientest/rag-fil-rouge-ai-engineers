"""Generate an answer from the passages found, citing the fiches used."""
import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from src.ingestion import load_sections
from src.retrieval import expand_to_sections, search

load_dotenv()

SYSTEM_PROMPT = """Vous répondez aux questions de salariés du secteur privé sur leurs congés.

Règles :
- Répondez uniquement à partir des extraits fournis, en français, de façon concise.
- Citez dans "sources" l'identifiant de chaque fiche que vous avez utilisée, par exemple F2381.
- Si les extraits ne permettent pas de répondre avec certitude, mettez exactement NO_ANSWER dans "answer" et une liste vide dans "sources"."""


def build_context(docs: list[Document]) -> str:
    """Put the passages in one text: the fiche id in brackets, then the passage."""
    parts = []
    for doc in docs:
        parts.append("[" + doc.metadata["doc_id"] + "]\n" + doc.page_content)
    return "\n\n".join(parts)


class Answer(BaseModel):
    """The answer of the assistant."""

    answer: str = Field(description="Réponse à la question, ou NO_ANSWER si les extraits ne permettent pas de répondre")
    sources: list[str] = Field(description="Identifiants des fiches utilisées, par exemple F2381 ; liste vide si NO_ANSWER")


@traceable(name="generate_answer")
def generate_answer(question: str, context: str) -> Answer:
    """Ask the model to answer the question from the context, as an Answer object."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage("Extraits :\n\n" + context + "\n\nQuestion : " + question)]
    return llm.with_structured_output(Answer).invoke(messages)


@traceable(name="answer_question", process_inputs=lambda inputs: {"question": inputs["question"]})
def answer_question(question: str, k: int = 5, sections: dict | None = None) -> tuple[Answer, list[Document]]:
    """Search the passages, give them to the model, and return the answer with the passages used."""
    if sections is None:
        sections = load_sections()
    docs = expand_to_sections(search(question, k=k, collection="chunks_titled"), sections)
    return generate_answer(question, build_context(docs)), docs


def check_sources(answer: Answer, docs: list[Document]) -> list[str]:
    """Return the cited fiches that were not in the passages given to the model (invented sources)."""
    given = {doc.metadata["doc_id"] for doc in docs}
    return [source for source in answer.sources if source not in given]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Combien de jours de congé pour mon mariage ?"
    answer, docs = answer_question(question)
    print(answer.answer)
    print("Sources :", ", ".join(answer.sources) or "aucune")
