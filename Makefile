.PHONY: help install env check notebook download corpus pipeline parse index index-titled ask agent eval traces test test-all mcp

ID ?= F33902
QUESTION ?= Combien de jours de congé pour mon mariage ?
VERSION ?= titled

help: ## Liste les commandes
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

install: ## Installe les dépendances (uv sync)
	uv sync

env: ## Crée .env depuis .env.example s'il n'existe pas
	@test -f .env || cp .env.example .env

check: ## Vérifie que l'API répond (modèle de chat et embedder)
	uv run python check_gateway.py

notebook: ## Ouvre Jupyter dans le navigateur (dossier notebooks/)
	uv run jupyter notebook notebooks/

download: ## Télécharge et décompresse le XML Service-Public dans data/raw/
	mkdir -p data/raw
	curl -L -z data/raw/vosdroits-latest.zip -o data/raw/vosdroits-latest.zip https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip
	unzip -q -o data/raw/vosdroits-latest.zip -d data/raw/xml

corpus: ## Génère data/corpus/*.md depuis les fiches XML (après make download)
	uv run python -m src.pipeline corpus

pipeline: download ## Télécharge les fiches, écrit le corpus, découpe en chunks et affiche les tailles
	uv run python -m src.pipeline

parse: ## Convertit une fiche XML en Markdown et l'affiche (ID=F33902)
	uv run python -m src.parsing $(ID)

index: ## Construit le store Qdrant local (collection chunks_500)
	uv run python -c "from src.ingestion import load_chunks; from src.retrieval import build_store; build_store(load_chunks(), 'chunks_500')"

index-titled: ## Construit le store Qdrant avec le titre devant chaque chunk (collection chunks_titled)
	uv run python -c "from src.ingestion import load_chunks; from src.retrieval import build_store; build_store(load_chunks(with_title=True), 'chunks_titled')"

ask: ## Pose une question au RAG (QUESTION="...")
	uv run python -m src.generation "$(QUESTION)"

agent: ## Pose une question à l'agent (QUESTION="...")
	uv run python -m src.agent "$(QUESTION)"

eval: ## Mesure une version du RAG sur les 30 questions (VERSION=baseline|titled)
	uv run python -m src.evaluation $(VERSION)

traces: ## Résume les traces LangSmith (latence, tokens, coût) ; demande une clé LangSmith
	uv run python -m src.monitoring

test: ## Tests rapides : sans modèle ni réseau
	uv run pytest -m "not network"

test-all: ## Tous les tests, dont la non-régression de la recherche (API + index Qdrant)
	uv run pytest

mcp: ## Lance le serveur MCP du RAG (stdio) ; à brancher dans un client MCP
	uv run python -m src.mcp_server
