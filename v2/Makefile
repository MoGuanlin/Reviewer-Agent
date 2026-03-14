PYTHON ?= python
SCRIPT := gemini.py

PDF ?=
TITLE ?=
MODEL ?=
PROMPTS ?=
OUTDIR ?= outputs

SMOKE_ARGS :=
PDF_ARGS :=
REVIEW_ARGS := --outdir $(OUTDIR)

ifneq ($(strip $(MODEL)),)
SMOKE_ARGS += --model $(MODEL)
PDF_ARGS += --model $(MODEL)
REVIEW_ARGS += --model $(MODEL)
endif

ifneq ($(strip $(TITLE)),)
REVIEW_ARGS += --title "$(TITLE)"
endif

ifneq ($(strip $(PROMPTS)),)
REVIEW_ARGS += --prompts "$(PROMPTS)"
endif

.PHONY: help smoke pdf review

help:
	@echo "Usage:"
	@echo "  make smoke [MODEL=gemini-3.1-pro-preview]"
	@echo "  make pdf PDF=paper.pdf [MODEL=...]"
	@echo "  make review PDF=paper.pdf [TITLE=...] [MODEL=...] [PROMPTS=...] [OUTDIR=outputs]"

smoke:
	$(PYTHON) $(SCRIPT) smoke $(SMOKE_ARGS)

pdf:
	@if [ -z "$(PDF)" ]; then echo "Missing PDF. Example: make pdf PDF=9412.pdf"; exit 1; fi
	$(PYTHON) $(SCRIPT) pdf "$(PDF)" $(PDF_ARGS)

review:
	@if [ -z "$(PDF)" ]; then echo "Missing PDF. Example: make review PDF=9412.pdf"; exit 1; fi
	$(PYTHON) $(SCRIPT) review "$(PDF)" $(REVIEW_ARGS)
