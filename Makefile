SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

DOCKER_COMPOSE_FILE ?= docker-compose.yml
DOCKER_PROJECT ?= agentic-rag
ENV_FILE ?= .env

.PHONY: help test lint typecheck poetry-check diff-check check ensure-env docker-build docker-up docker-up-build docker-up-recreate docker-down docker-stop docker-logs docker-ps docker-restart docker-exec docker-smoke-embedding-worker

help:
	@printf '%s\n' "Available targets:"
	@printf '%s\n' "  make test          Run pytest"
	@printf '%s\n' "  make lint          Run ruff checks"
	@printf '%s\n' "  make typecheck     Run mypy on src"
	@printf '%s\n' "  make poetry-check  Validate Poetry project metadata"
	@printf '%s\n' "  make diff-check    Check git diff for whitespace errors"
	@printf '%s\n' "  make check         Run all validation checks"
	@printf '%s\n' "  make docker-build  Build local Docker images"
	@printf '%s\n' "  make docker-up     Start local Docker stack"
	@printf '%s\n' "  make docker-up-build Build and start local Docker stack"
	@printf '%s\n' "  make docker-up-recreate Recreate local Docker stack"
	@printf '%s\n' "  make docker-down   Stop local Docker stack"
	@printf '%s\n' "  make docker-stop   Stop local Docker stack without removing containers"
	@printf '%s\n' "  make docker-logs   Follow local Docker logs"
	@printf '%s\n' "  make docker-ps     Show local Docker services"
	@printf '%s\n' "  make docker-restart Restart local Docker services"
	@printf '%s\n' "  make docker-exec SERVICE=api Open a shell in a service"
	@printf '%s\n' "  make docker-smoke-embedding-worker Check embedding worker container imports and DB connection"

test:
	poetry run pytest

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

poetry-check:
	poetry check

diff-check:
	git diff --check

check: test lint typecheck poetry-check diff-check

ensure-env:
	@test -f $(ENV_FILE) || (printf '%s\n' "Missing $(ENV_FILE). Run: cp .env.template $(ENV_FILE)" && exit 1)

docker-build: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) build

docker-up: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) up -d

docker-up-build: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) up -d --build

docker-up-recreate: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) up -d --force-recreate

docker-down: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) down -v

docker-stop: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) stop

docker-logs: ensure-env
ifeq ($(SERVICE),)
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) logs -f --tail=100
else
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) logs -f --tail=100 $(SERVICE)
endif

docker-ps: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) ps

docker-restart: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) restart

docker-exec: ensure-env
ifndef SERVICE
	$(error Please provide SERVICE name. Usage: make docker-exec SERVICE=api)
endif
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) exec $(SERVICE) bash

docker-smoke-embedding-worker: ensure-env
	docker compose --env-file $(ENV_FILE) -f $(DOCKER_COMPOSE_FILE) -p $(DOCKER_PROJECT) exec embedding-worker python -c 'from sqlalchemy import text; from agentic_rag.shared.db.session import get_sync_session_factory; from agentic_rag.workers.embedding import process_embedding_batches; SessionLocal = get_sync_session_factory(); session = SessionLocal(); session.execute(text("SELECT 1")); session.close(); print("embedding-worker smoke ok")'
