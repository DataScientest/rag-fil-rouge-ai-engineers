# RAG fil rouge — congés dans le secteur privé

Le projet complet du parcours AI Engineers : un assistant qui répond aux questions sur les congés du secteur privé à partir de 55 fiches Service-Public, en citant ses sources. Vous y retrouvez le code de chaque étape vue en formation, prêt à relancer chez vous.

## Ce qu'il faut

- Python 3.12 et [uv](https://docs.astral.sh/uv/getting-started/installation/)
- une clé d'API compatible OpenAI (OpenAI directement, ou une passerelle comme celle de la formation). Le code utilise `gpt-4o-mini` pour le chat et `text-embedding-3-small` pour les embeddings.
- facultatif : une clé [LangSmith](https://smith.langchain.com) pour les traces (`make traces`)

## Démarrer

```bash
make install   # installe les dépendances
make env       # crée .env depuis .env.example
```

Renseigner `.env` :

```bash
LLM_BASE_URL=https://api.openai.com/v1   # ou l'adresse de votre passerelle
LLM_API_KEY=sk-...
```

Puis :

```bash
make check          # l'API répond : « Prêt. » et « 1536 numbers »
make test           # tests rapides, sans réseau
make index-titled   # construit l'index Qdrant local depuis data/corpus/
make ask QUESTION="Combien de jours de congé pour mon mariage ?"
```

`make help` liste toutes les commandes.

## Le projet, étape par étape

| Étape | Fichier | Commande |
|---|---|---|
| Lire une fiche XML et la convertir en Markdown | `src/parsing.py` | `make parse ID=F33902` |
| Construire le corpus depuis le XML officiel | `src/pipeline.py` | `make download` puis `make corpus` |
| Découper les fiches en chunks | `src/ingestion.py` | |
| Indexer et rechercher (vectoriel, BM25, hybride) | `src/retrieval.py` | `make index`, `make index-titled` |
| Générer une réponse sourcée | `src/generation.py` | `make ask` |
| Évaluer sur 30 questions de référence | `src/evaluation.py` | `make eval VERSION=titled` |
| Agent avec outils | `src/agent.py` | `make agent` |
| Suivre coûts et latence (LangSmith) | `src/monitoring.py` | `make traces` |
| Exposer le RAG en serveur MCP | `src/mcp_server.py` | `make mcp` |

Les notebooks (`make notebook`) : `01_discover.ipynb` explore les fiches XML brutes (lancer `make download` avant), `explore.ipynb` appelle les fonctions de `src/` pas à pas.

## Bon à savoir

- Le corpus (`data/corpus/`, 55 fiches) et le jeu d'évaluation (`data/eval/ref_v2.jsonl`) sont fournis : rien à télécharger pour l'index et l'évaluation.
- Qdrant local n'accepte qu'un processus à la fois : fermez le notebook avant de lancer `make ask` ou `make eval` sur le même index.
- Les appels au modèle sont payants : `make eval` pose 30 questions et appelle un juge pour chacune.
- `make eval` écrit ses résultats dans `data/eval/runs/`.

## Données

Fiches pratiques Service-Public.gouv.fr (DILA), dossier N510 « Congés dans le secteur privé », sous Licence Ouverte 2.0 (Etalab). Détails dans `data/README.md`.
