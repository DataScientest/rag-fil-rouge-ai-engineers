"""Regression tests of the RAG.

Fast tests: no model, no network, deterministic.
Network tests (marker `network`): search the real Qdrant index, so they embed the questions through the gateway.
"""
import asyncio
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.evaluation import (evidence_in_context, fiches_found, is_refusal, load_reference, normalize, summarize)
from src.generation import Answer, build_context, check_sources
from src.ingestion import load_chunks, load_sections, section_key
from src.parsing import fiche_to_markdown
from src.retrieval import expand_to_sections, tokenize

DUMMY_KEY = "unused-in-fast-tests"  # set by conftest.py when no real key exists

# Thresholds = the values measured for the "titled" version (chapter 7). A drop below them fails the build.
MIN_EVIDENCE_IN_CONTEXT = 15  # out of the 22 answerable questions
MIN_FICHES_FOUND = 18


def doc(doc_id: str, text: str, section_path: str = "Fiche > Section") -> Document:
    """A passage with the metadata of a real chunk."""
    return Document(page_content=text, metadata={"doc_id": doc_id, "section_path": section_path})


# ---------- fast tests ----------

FICHE_XML = """<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F1" spUrl="https://example.org/F1">
  <dc:title>Congé de test</dc:title>
  <dc:date>modified 2026-01-02</dc:date>
  <Introduction><Texte><Paragraphe>Un salarié peut s'absenter.</Paragraphe></Texte></Introduction>
  <Texte><Chapitre><Titre>Durée</Titre><Paragraphe>Elle est de 4 jours (<LienInterne LienPublication="F2">voir</LienInterne>).</Paragraphe></Chapitre></Texte>
</Publication>"""


def test_fiche_to_markdown_keeps_title_source_sections_and_links():
    markdown = fiche_to_markdown(ET.fromstring(FICHE_XML), {"F1", "F2"})
    assert markdown.startswith("# Congé de test")
    assert "Source : https://example.org/F1 (mise à jour : 2026-01-02)" in markdown
    assert "## Durée" in markdown
    assert "(voir F2)" in markdown


def test_normalize_ignores_case_apostrophes_and_spaces():
    assert normalize("  L’employeur   DOIT ") == "l'employeur doit"


def test_evidence_in_context_requires_every_phrase():
    item = {"evidence": "La durée est de 4 jours. [...] Les jours sont payés normalement."}
    both = {"context": "La durée est de 4 jours.\n\nLes jours sont payés  normalement."}
    only_first = {"context": "La durée est de 4 jours."}
    assert evidence_in_context(item, both)
    assert not evidence_in_context(item, only_first)


def test_evidence_in_context_splits_table_cells_and_ignores_short_phrases():
    item = {"evidence": "Père, mère, beau-père ou belle-mère | 3 jours ou durée plus élevée"}
    assert evidence_in_context(item, {"context": "père, mère, beau-père ou belle-mère ... 3 jours ou durée plus élevée"})
    assert evidence_in_context({"evidence": "ok [...] court"}, {"context": "ok"})  # phrases of 15 characters or less are ignored


def test_fiches_found_needs_all_gold_fiches():
    item = {"gold_doc_ids": ["F1", "F2"]}
    assert fiches_found(item, {"doc_ids": ["F1", "F2", "F3"]})
    assert not fiches_found(item, {"doc_ids": ["F1"]})


def test_is_refusal():
    assert is_refusal({"answer": " NO_ANSWER\n"})
    assert not is_refusal({"answer": "4 jours"})


def test_check_sources_returns_invented_sources_only():
    docs = [doc("F1", "a"), doc("F2", "b")]
    assert check_sources(Answer(answer="x", sources=["F1", "F9"]), docs) == ["F9"]
    assert check_sources(Answer(answer="x", sources=["F1", "F2"]), docs) == []


def test_build_context_puts_the_fiche_id_before_each_passage():
    assert build_context([doc("F1", "texte un"), doc("F2", "texte deux")]) == "[F1]\ntexte un\n\n[F2]\ntexte deux"


def test_tokenize_keeps_article_numbers_in_one_word():
    assert tokenize("Article L3142-4 du Code") == ["article", "l3142-4", "du", "code"]


def test_expand_to_sections_replaces_chunks_and_removes_duplicates():
    key = section_key("F1", "Fiche > Section")
    sections = {key: "[Fiche > Section]\ntoute la section"}
    docs = [doc("F1", "morceau 1", "Fiche > Section > A"), doc("F1", "morceau 2", "Fiche > Section > B")]
    expanded = expand_to_sections(docs, sections)
    assert [d.page_content for d in expanded] == ["[Fiche > Section]\ntoute la section"]


def test_expand_to_sections_keeps_the_chunk_when_the_section_is_too_long():
    key = section_key("F1", "Fiche > Section")
    expanded = expand_to_sections([doc("F1", "morceau")], {key: "x" * 50}, max_chars=10)
    assert expanded[0].page_content == "morceau"


def test_summarize_counts_a_fake_run():
    questions = [
        {"id": "q1", "type": "simple", "gold_doc_ids": ["F1"], "evidence": "la preuve est dans le contexte"},
        {"id": "q2", "type": "simple", "gold_doc_ids": ["F2"], "evidence": "une preuve qui manque ici"},
        {"id": "q3", "type": "unanswerable", "gold_doc_ids": [], "evidence": ""},
    ]
    results = [
        {"answer": "ok", "doc_ids": ["F1"], "context": "la preuve est dans le contexte", "verdict": "correct"},
        {"answer": "NO_ANSWER", "doc_ids": ["F3"], "context": "rien", "verdict": "refus"},
        {"answer": "NO_ANSWER", "doc_ids": [], "context": "", "verdict": "refus"},
    ]
    assert summarize(questions, results) == {"fiches_found": 1, "evidence_in_context": 1, "correct": 1, "partial": 0,
                                             "wrong": 0, "wrongly_refused": 1, "refused": 1}


def test_reference_set_is_well_formed():
    questions = load_reference()
    assert len(questions) == 30
    assert len({q["id"] for q in questions}) == 30
    assert sum(q["type"] != "unanswerable" for q in questions) == 22
    for q in questions:
        assert q["question"].strip()


def test_gold_fiches_exist_in_the_corpus():
    sections = load_sections()
    fiches_in_corpus = {key.split(" | ")[0] for key in sections}
    for q in load_reference():
        assert set(q["gold_doc_ids"]) <= fiches_in_corpus, q["id"]


def test_answerable_evidence_is_in_the_corpus():
    """Each expected evidence exists in the text of the right fiches: otherwise the reference set itself is wrong."""
    for q in load_reference():
        if q["type"] == "unanswerable":
            continue
        text = "\n".join(Path("data/corpus/" + doc_id + ".md").read_text(encoding="utf-8") for doc_id in q["gold_doc_ids"])
        assert evidence_in_context(q, {"context": text}), q["id"]


def test_chunks_with_title_start_with_their_section_path():
    chunks = load_chunks(with_title=True)[:50]
    for chunk in chunks:
        assert chunk.page_content.startswith("[" + chunk.metadata["section_path"] + "]")


def test_mcp_server_exposes_the_two_tools():
    from src.mcp_server import mcp

    tools = asyncio.run(mcp.list_tools())
    assert {tool.name for tool in tools} == {"search_fiches", "ask_fiches"}


# ---------- network tests: real index, embeddings through the gateway ----------

@pytest.fixture(scope="session")
def titled_run():
    """Search the 'titled' version for every question (retrieval only, no answer generated)."""
    if not Path("data/qdrant").exists():
        pytest.skip("index data/qdrant missing: run `make index-titled`")
    from src.retrieval import search

    if os.environ["LLM_API_KEY"] == DUMMY_KEY:
        pytest.skip("no gateway key: fill .env or LLM_API_KEY")
    sections = load_sections()
    questions = load_reference()
    results = []
    for q in questions:
        docs = expand_to_sections(search(q["question"], k=5, collection="chunks_titled"), sections)
        results.append({"doc_ids": sorted({d.metadata["doc_id"] for d in docs}), "context": build_context(docs)})
    return questions, results


@pytest.mark.network
def test_evidence_in_context_does_not_drop(titled_run):
    questions, results = titled_run
    found = sum(evidence_in_context(q, r) for q, r in zip(questions, results) if q["type"] != "unanswerable")
    assert found >= MIN_EVIDENCE_IN_CONTEXT, f"evidence in context: {found}/22"


@pytest.mark.network
def test_fiches_found_does_not_drop(titled_run):
    questions, results = titled_run
    found = sum(fiches_found(q, r) for q, r in zip(questions, results) if q["type"] != "unanswerable")
    assert found >= MIN_FICHES_FOUND, f"fiches found: {found}/22"


@pytest.mark.network
def test_search_finds_the_marriage_fiche_first():
    from src.retrieval import search

    docs = search("Combien de jours de congé pour mon mariage ?", k=3, collection="chunks_titled")
    assert docs[0].metadata["doc_id"] == "F34154"
