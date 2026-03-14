# Makefile for Gemini Reviewer

# ==========================================
# Configuration Variables
# ==========================================

# 1. Input PDF path
PDF ?= docs/31489_Sharp_Inequalities_betwe.pdf

# 2. Output directory (default: based on PDF filename stem)
#    If you want to manually set it: make review OUTDIR=custom_output
OUTDIR ?= outputs/$(basename $(notdir $(PDF)))_review

# 3. Model selection
MODEL ?= gemini-3.1-pro-preview

# 4. Prompt file selection
PROMPTS ?= src/reviewers/prompts_comprehensive.py

# 5. Optional title override
#    Usage: make review TITLE="My Custom Title"
TITLE ?= 

# ==========================================
# Internal Variables
# ==========================================

PYTHON := python
SCRIPT := src/reviewers/gemini.py
FLAGS := 

ifneq ($(TITLE),)
	FLAGS += --title "$(TITLE)"
endif

# ==========================================
# Targets
# ==========================================

.PHONY: help review smoke pdf clean

help:
	@echo "Gemini Reviewer Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make review                Run the full review pipeline with defaults."
	@echo "  make review PDF=path/to.pdf   Run review on a specific PDF."
	@echo "  make review MODEL=gemini-1.5-pro   Run with a different model."
	@echo "  make smoke                 Run a basic API connectivity test."
	@echo "  make pdf PDF=path/to.pdf   Run a quick PDF upload test."
	@echo "  make clean                 Remove the default outputs directory."
	@echo ""
	@echo "Current Configuration:"
	@echo "  PDF     = $(PDF)"
	@echo "  OUTDIR  = $(OUTDIR)"
	@echo "  MODEL   = $(MODEL)"
	@echo "  PROMPTS = $(PROMPTS)"

review:
	@echo "Starting review for: $(PDF)"
	@echo "Model: $(MODEL)"
	@echo "Output: $(OUTDIR)"
	@mkdir -p $(OUTDIR)
	$(PYTHON) $(SCRIPT) review \
		$(PDF) \
		--prompts $(PROMPTS) \
		--model $(MODEL) \
		--outdir $(OUTDIR) \
		$(FLAGS)

smoke:
	@echo "Running smoke test..."
	$(PYTHON) $(SCRIPT) smoke --model $(MODEL)

pdf:
	@echo "Running PDF smoke test..."
	$(PYTHON) $(SCRIPT) pdf $(PDF) --model $(MODEL)

clean:
	rm -rf outputs/
