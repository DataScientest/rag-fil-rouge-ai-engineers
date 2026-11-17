"""Measure the assistant on a reference set: right fiches found, refusals, correct answers, noise."""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from src.generation import build_context, generate_answer
from src.ingestion import load_sections
from src.retrieval import expand_to_sections, search

load_dotenv()

REFERENCE_PATH = "data/eval/ref_v2.jsonl"
RUNS_DIR = Path("data/eval/runs")


def load_reference(path: str = REFERENCE_PATH) -> list[dict]:
    """Read the reference set: one question per line, with its expected fiches and answer."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def run_baseline(question: str, sections: dict) -> tuple[str, list[str], list[Document]]:
    """Version 1: the 500-token chunks without titles, given to the model as they are."""
    docs = search(question, k=5, collection="chunks_500")
    answer = generate_answer(question, build_context(docs))
    return answer.answer, answer.sources, docs


def run_titled(question: str, sections: dict) -> tuple[str, list[str], list[Document]]:
    """Version 2: chunks with the title in the index, replaced by their whole section."""
    docs = expand_to_sections(search(question, k=5, collection="chunks_titled"), sections)
    answer = generate_answer(question, build_context(docs))
    return answer.answer, answer.sources, docs


def run_agent(question: str, sections: dict) -> tuple[str, list[str], list[Document]]:
    """Version 3: an agent that searches and reads the fiches by itself."""
    from src.agent import ask_agent  # imported here: src.agent loads the sections as soon as it is imported

    answer, docs, calls = ask_agent(question)
    return answer.answer, answer.sources, docs


VERSIONS = {"baseline": run_baseline, "titled": run_titled, "agent": run_agent}


@traceable(name="rag_question", process_inputs=lambda inputs: {"version": inputs["version"], "question": inputs["question"]})
def ask_version(version: str, question: str, sections: dict) -> tuple[str, list[str], list[Document]]:
    """Ask one question to one version: one call is one trace in LangSmith."""
    return VERSIONS[version](question, sections)


def run_set(questions: list[dict], version: str, sections: dict) -> list[dict]:
    """Ask every question to one version and keep everything needed to audit the run later."""
    results = []
    for item in questions:
        answer, sources, docs = ask_version(version, item["question"], sections,
                                            langsmith_extra={"tags": [version], "metadata": {"question_id": item["id"]}})
        results.append({
            "id": item["id"],
            "answer": answer,
            "sources": sources,
            "doc_ids": sorted({doc.metadata["doc_id"] for doc in docs}),
            "context": build_context(docs),
        })
    return results


def save_run(results: list[dict], name: str) -> Path:
    """Write a run to data/eval/runs/<name>.jsonl, one result per line."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / (name + ".jsonl")
    with open(path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    return path


def load_run(name: str) -> list[dict]:
    """Read a run saved by save_run."""
    with open(RUNS_DIR / (name + ".jsonl"), encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def fiches_found(item: dict, result: dict) -> bool:
    """True if all the expected fiches were in the passages given to the model."""
    return set(item["gold_doc_ids"]) <= set(result["doc_ids"])


def normalize(text: str) -> str:
    """Lowercase, straighten apostrophes and collapse spaces, so that two spellings of a phrase match."""
    return " ".join(text.lower().replace("’", "'").split())


def evidence_in_context(item: dict, result: dict) -> bool:
    """True if every phrase of the evidence is in the text given to the model.

    The evidence of a question is a few phrases joined by [...]; table cells are split on |.
    """
    context = normalize(result["context"])
    for part in item["evidence"].replace("|", "[...]").split("[...]"):
        phrase = normalize(part)
        if len(phrase) > 15 and phrase not in context:
            return False
    return True


def is_refusal(result: dict) -> bool:
    """True if the assistant refused to answer."""
    return result["answer"].strip() == "NO_ANSWER"


class Grade(BaseModel):
    """The verdict of the grader on one answer."""

    asked_facts: list[str] = Field(description="Les faits que la question demande, tirés de la réponse attendue (pas les précisions secondaires)")
    missing_or_wrong: list[str] = Field(description="Parmi ces faits, ceux qui manquent dans la réponse de l'assistant ou qu'elle contredit ; liste vide si aucun")
    verdict: Literal["correct", "partiel", "faux"] = Field(
        description="correct si aucun fait demandé ne manque ; partiel si un fait manque ; faux si un fait est contredit")


GRADER_PROMPT = """Vous corrigez les réponses d'un assistant sur les congés des salariés. On vous donne la question, la réponse attendue (juste) et la réponse de l'assistant.

Étape 1 : listez les faits que la question demande (courts). Ignorez les précisions secondaires de la réponse attendue que la question ne demande pas.
Étape 2 : pour chaque fait, vérifiez s'il est dans la réponse de l'assistant, sans contradiction.
Étape 3 : donnez le verdict."""


def grade_answer(item: dict, answer: str) -> Grade:
    """Ask a model to compare an answer with the expected answer."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    text = ("Question : " + item["question"] + "\n\nRéponse attendue : " + item["reference_answer"]
            + "\n\nRéponse de l'assistant : " + answer)
    return llm.with_structured_output(Grade).invoke([SystemMessage(GRADER_PROMPT), HumanMessage(text)])


def verdict_of(item: dict, result: dict) -> str:
    """Give one verdict to an answer: 'refus', or 'correct', 'partiel', 'faux'."""
    if is_refusal(result):
        return "refus"
    if item["type"] == "unanswerable":
        return "faux"
    return grade_answer(item, result["answer"]).verdict


def add_verdicts(questions: list[dict], results: list[dict]) -> None:
    """Write the verdict of every answer in its result."""
    for item, result in zip(questions, results):
        result["verdict"] = verdict_of(item, result)


def summarize(questions: list[dict], results: list[dict]) -> dict:
    """Count, for one run: fiches found, evidence in context, refusals, and verdicts."""
    answerable = [(q, r) for q, r in zip(questions, results) if q["type"] != "unanswerable"]
    unanswerable = [(q, r) for q, r in zip(questions, results) if q["type"] == "unanswerable"]
    verdicts = Counter(r["verdict"] for _, r in answerable)
    return {
        "fiches_found": sum(fiches_found(q, r) for q, r in answerable),
        "evidence_in_context": sum(evidence_in_context(q, r) for q, r in answerable),
        "correct": verdicts["correct"],
        "partial": verdicts["partiel"],
        "wrong": verdicts["faux"],
        "wrongly_refused": verdicts["refus"],
        "refused": sum(is_refusal(r) for _, r in unanswerable),
    }


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "titled"
    name = sys.argv[2] if len(sys.argv) > 2 else version
    questions = load_reference()
    results = run_set(questions, version, load_sections())
    add_verdicts(questions, results)
    save_run(results, name)
    print(version, summarize(questions, results))
