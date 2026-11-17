"""Build the dataset: XML fiches -> Markdown corpus -> chunks."""
import sys
from pathlib import Path

import tiktoken

from src.ingestion import load_chunks
from src.parsing import fiche_to_markdown, list_fiches

XML_DIR = "data/raw/xml"
CORPUS_DIR = "data/corpus"


def build_corpus(dossier_id: str = "N510") -> int:
    """Write one Markdown file per fiche of the dossier into data/corpus/ and return the number of files."""
    fiches = list_fiches(XML_DIR, dossier_id)
    Path(CORPUS_DIR).mkdir(parents=True, exist_ok=True)
    for fiche_id, root in fiches.items():
        text = fiche_to_markdown(root, set(fiches))
        Path(CORPUS_DIR, fiche_id + ".md").write_text(text, encoding="utf-8")
    return len(fiches)


def describe_chunks(chunks: list) -> None:
    """Print the number of chunks and their size in tokens."""
    encoder = tiktoken.get_encoding("cl100k_base")
    sizes = [len(encoder.encode(chunk.page_content)) for chunk in chunks]
    print(len(chunks), "chunks")
    print("taille moyenne :", sum(sizes) // len(sizes), "tokens | maximum :", max(sizes), "tokens")


def main(step: str = "all", dossier_id: str = "N510") -> None:
    """Run one step of the pipeline ("corpus" or "chunks"), or both ("all")."""
    if step in ("corpus", "all"):
        print(build_corpus(dossier_id), "fiches écrites dans", CORPUS_DIR)
    if step in ("chunks", "all"):
        describe_chunks(load_chunks(CORPUS_DIR, with_title=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
