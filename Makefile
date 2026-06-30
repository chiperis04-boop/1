.DEFAULT_GOAL := help
PY ?= python
VENV ?= .venv

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

setup: ## Create venv + install deps (CPU torch). For GPU, see docs/SETUP.md.
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && \
		pip install torch torchvision && \
		pip install -r requirements.txt
	@echo ">> Activate with: source $(VENV)/bin/activate"

bootstrap: ## Auto-install EVERYTHING (torch GPU/CPU autodetect + deps + all model weights)
	$(PY) -m scripts.setup

models: ## Download / prepare models
	bash scripts/download_models.sh

detect: ## Detect-only on a match: make detect IN=input/match.mp4
	. $(VENV)/bin/activate && $(PY) -m src.pipeline detect $(IN)

run: ## Full pipeline: make run IN=input/match.mp4 PROFILE=tiktok
	. $(VENV)/bin/activate && $(PY) -m src.pipeline run $(IN) --profile $(or $(PROFILE),tiktok)

profiles: ## List output profiles
	. $(VENV)/bin/activate && $(PY) -m src.pipeline list-profiles

check: ## Byte-compile all sources (fast sanity check)
	$(PY) -m compileall -q src

clean: ## Remove work/output artifacts
	rm -rf output/*/work

.PHONY: help setup models detect run profiles check clean bootstrap
