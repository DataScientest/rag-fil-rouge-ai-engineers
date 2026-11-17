"""Local Qdrant index (data/qdrant/) and dense top-k search, with optional improvements (section expansion, sub-questions, hybrid search)."""
import os
import re

import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.ingestion import section_key

load_dotenv()
QDRANT_PATH = "data/qdrant"
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small", base_url=os.environ["LLM_BASE_URL"],
                              api_key=os.environ["LLM_API_KEY"], check_embedding_ctx_length=False)


def build_store(chunks: list[Document], collection: str = "chunks_500") -> None:
    """Embed the chunks and write them to a Qdrant collection on disk (once per configuration)."""
    store = QdrantVectorStore.from_documents(chunks, EMBEDDINGS, path=QDRANT_PATH,
                                             collection_name=collection, force_recreate=True)
    store.client.close()  # frees data/qdrant/: only one client at a time in local mode


@traceable(run_type="retriever", name="search")
def search(question: str, k: int = 5, collection: str = "chunks_500") -> list[Document]:
    """Return the k chunks closest to the question, reopening the index without re-embedding."""
    store = QdrantVectorStore.from_existing_collection(collection, EMBEDDINGS, path=QDRANT_PATH)
    docs = store.similarity_search(question, k=k)
    store.client.close()
    return docs


def expand_to_sections(docs: list[Document], sections: dict[str, str], max_chars: int = 6000) -> list[Document]:
    """Replace each found chunk by the text of its whole section, without duplicates.

    A section longer than max_chars characters is too long: the found chunk is kept instead.
    `sections` comes from load_sections().
    """
    expanded = []
    seen = set()
    for doc in docs:
        key = section_key(doc.metadata["doc_id"], doc.metadata["section_path"])
        text = sections.get(key)
        if text is None or len(text) > max_chars:
            text = doc.page_content
            key = text
        if key in seen:
            continue
        seen.add(key)
        expanded.append(Document(page_content=text, metadata=doc.metadata))
    return expanded


class SubQuestions(BaseModel):
    """The sub-questions of a question."""

    sub_questions: list[str] = Field(description="1 to 3 standalone questions, each looking for ONE piece of information")


def split_question(question: str) -> list[str]:
    """Ask the model to cut a question into 1 to 3 standalone sub-questions."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    splitter = llm.with_structured_output(SubQuestions)
    prompt = ("Un salarié pose une question sur le droit du travail. Découpez-la en 1 à 3 sous-questions autonomes, "
              "chacune cherchant UNE seule information dans une fiche pratique. "
              "Si la question est simple, renvoyez-la telle quelle.\n\nQuestion : " + question)
    return splitter.invoke(prompt).sub_questions[:3]


def search_multi(question: str, k: int = 5, collection: str = "chunks_titled") -> list[Document]:
    """Search the question and its sub-questions, then take one result of each list in turn."""
    queries = [question] + split_question(question)
    ranked_lists = [search(query, k=2 * k, collection=collection) for query in queries]
    merged = []
    seen = set()
    for rank in range(2 * k):
        for docs in ranked_lists:
            doc = docs[rank]
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
    return merged[:k]


def tokenize(text: str) -> list[str]:
    """Lowercase words; hyphens are kept, so that "L3142-4" stays one word."""
    return re.findall(r"[\w-]+", text.lower())


def build_bm25(chunks: list[Document]) -> BM25Okapi:
    """Index the chunks for the keyword search (BM25)."""
    return BM25Okapi([tokenize(chunk.page_content) for chunk in chunks])


def search_hybrid(question: str, chunks: list[Document], bm25: BM25Okapi, k: int = 5,
                  collection: str = "chunks_titled") -> list[Document]:
    """Merge the dense ranking and the BM25 ranking with reciprocal rank fusion (RRF)."""
    dense_docs = search(question, k=50, collection=collection)
    scores = bm25.get_scores(tokenize(question))
    bm25_docs = [chunks[i] for i in np.argsort(-scores)[:50]]
    fused = {}
    docs_by_text = {}
    for ranking in (dense_docs, bm25_docs):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc.page_content] = fused.get(doc.page_content, 0) + 1 / (60 + rank)
            docs_by_text[doc.page_content] = doc
    best = sorted(fused, key=lambda text: -fused[text])[:k]
    return [docs_by_text[text] for text in best]
