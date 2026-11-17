"""Split the corpus fiches into LangChain chunks, with doc_id, section_path and refs as metadata."""
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "title"), ("##", "section"), ("###", "subsection")]


def make_section_path(metadata: dict) -> str:
    """Title path of a chunk, from the fiche down to the subsection: "Fiche > Section > Subsection"."""
    levels = []
    for _, key in HEADERS:
        if key in metadata:
            levels.append(metadata[key])
    return " > ".join(levels)


def load_chunks(corpus_dir: str = "data/corpus", chunk_size: int = 500, overlap: int = 0,
                with_title: bool = False) -> list[Document]:
    """Split each fiche on its titles, then into pieces of at most chunk_size tokens.

    with_title=True writes the title path in front of the chunk text: "[Fiche > Section]\\ntext".
    The title is then embedded with the text, and the model receives it too: it gives the chunk its context.
    """
    by_title = MarkdownHeaderTextSplitter(HEADERS)
    by_size = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        refs = sorted(set(re.findall(r"\b[FN]\d{3,6}\b", text)) - {path.stem})
        for chunk in by_size.split_documents(by_title.split_text(text)):
            section_path = make_section_path(chunk.metadata)
            if with_title:
                chunk.page_content = "[" + section_path + "]\n" + chunk.page_content
            chunk.metadata = {"doc_id": path.stem, "section_path": section_path, "refs": refs}
            chunks.append(chunk)
    return chunks


def section_key(doc_id: str, section_path: str) -> str:
    """Identifier of the section of a chunk: the fiche and the first two levels of its title path."""
    first_levels = section_path.split(" > ")[:2]
    return doc_id + " | " + " > ".join(first_levels)


def load_sections(corpus_dir: str = "data/corpus") -> dict[str, str]:
    """Full text of each section (the "##" level of a fiche), indexed by section_key().

    Each piece of the section keeps its title path in front, as with with_title=True.
    """
    by_title = MarkdownHeaderTextSplitter(HEADERS)
    sections = {}
    for path in sorted(Path(corpus_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for piece in by_title.split_text(text):
            section_path = make_section_path(piece.metadata)
            key = section_key(path.stem, section_path)
            block = "[" + section_path + "]\n" + piece.page_content.strip()
            if key in sections:
                sections[key] = sections[key] + "\n\n" + block
            else:
                sections[key] = block
    return sections
