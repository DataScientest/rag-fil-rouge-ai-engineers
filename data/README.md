# Corpus du projet

`corpus/` contient 55 fiches pratiques de Service-Public.gouv.fr, dossier N510 « Congés dans le secteur privé », converties en Markdown, une fiche par fichier. Le nom du fichier est le `doc_id`.

- Source : DILA, jeu de données « Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers », data.gouv.fr (https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip).
- Licence : Licence Ouverte 2.0 (Etalab).
- Date d'extraction : 29 septembre 2026.
- Conversion : `make corpus` (`src/pipeline.py`, fonctions de `src/parsing.py`). Les fiches « Recherche guidée » sont exclues. Seuls les renvois vers une fiche du corpus sont réécrits en « (voir F1234) » ; les autres liens restent du texte.
