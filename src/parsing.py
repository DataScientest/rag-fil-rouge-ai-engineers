"""Convert Service-Public fiches (XML) into Markdown text."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

DC = "{http://purl.org/dc/elements/1.1/}"
SECTION_TAGS = ("Chapitre", "SousChapitre", "Situation", "Cas")
CONTENT_TAGS = ("Introduction", "Texte", "ListeSituations")
IGNORED_TAGS = ("Titre", "TitreRiche", "Colonne")


def list_fiches(xml_dir: str = "data/raw/xml", dossier_id: str = "N510") -> dict[str, ET.Element]:
    """Read the fiches of one dossier, without the "Recherche guidée" ones."""
    fiches = {}
    for path in sorted(Path(xml_dir).glob("F*.xml")):
        root = ET.parse(path).getroot()
        in_dossier = root.find("DossierPere[@ID='" + dossier_id + "']") is not None
        if in_dossier and root.get("type") != "Recherche guidée":
            fiches[path.stem] = root
    return fiches


def element_text(element: ET.Element, known_ids: set[str]) -> str:
    """Text of an element on one line; links to a fiche of the project become "(voir F1234)"."""
    text = element.text or ""
    for child in element:
        text += element_text(child, known_ids)
        if child.tag == "LienInterne" and child.get("LienPublication") in known_ids:
            text += " (voir " + child.get("LienPublication") + ")"
        text += child.tail or ""
    return re.sub(r"\s+", " ", text)


def element_blocks(element: ET.Element, known_ids: set[str], level: int = 2) -> list[str]:
    """Markdown blocks of an element: section titles, paragraphs, list items and tables."""
    blocks = []
    for child in element:
        if child.tag in SECTION_TAGS:
            title = element_text(child.find("Titre"), known_ids).strip()
            blocks.append("\n" + "#" * min(level, 4) + " " + title + "\n")
            blocks += element_blocks(child, known_ids, level + 1)
        elif child.tag == "Paragraphe":
            blocks.append(element_text(child, known_ids).strip())
            blocks.append("")
        elif child.tag == "Item":
            blocks.append("- " + element_text(child, known_ids).strip())
        elif child.tag == "Rangée":
            cells = [element_text(cell, known_ids).strip() for cell in child]
            blocks.append("| " + " | ".join(cells) + " |")
        elif child.tag == "Tableau":
            rows = element_blocks(child, known_ids, level)
            columns = len(child.find("Rangée"))
            separator = "|" + " --- |" * columns
            blocks += ["", rows[0], separator] + rows[1:] + [""]
        elif child.tag not in IGNORED_TAGS:
            blocks += element_blocks(child, known_ids, level)
    return blocks


def fiche_to_markdown(root: ET.Element, known_ids: set[str]) -> str:
    """Markdown of a fiche: title, source, text, legal references and related questions."""
    body = []
    for child in root:
        if child.tag in CONTENT_TAGS:
            body += element_blocks(child, known_ids)

    references = []
    for reference in root.findall("Reference"):
        references.append("- " + element_text(reference.find("Titre"), known_ids).strip())

    related = []
    for question in root.findall("QuestionReponse"):
        if question.get("ID") in known_ids:
            related.append("- " + question.findtext("Titre") + " (voir " + question.get("ID") + ")")

    date = root.findtext(DC + "date")[9:]  # drops the word "modified "
    header = ["# " + root.findtext(DC + "title"), "", "Source : " + root.get("spUrl") + " (mise à jour : " + date + ")", ""]
    lines = header + body + ["\n## Références\n"] + references + ["\n## Questions ? Réponses !\n"] + related
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


if __name__ == "__main__":
    import sys

    fiche_id = sys.argv[1] if len(sys.argv) > 1 else "F33902"
    fiches = list_fiches()
    print(fiche_to_markdown(fiches[fiche_id], set(fiches)), end="")
