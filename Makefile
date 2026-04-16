# rustlab_llm — run lessons, render notebooks, clean generated files

SCRIPTS := $(sort $(wildcard lessons/*/*.r))
NOTEBOOKS := $(sort $(wildcard notebooks/*.md))
NOTEBOOK_HTMLS := $(patsubst notebooks/%.md,notebooks/outputs/%.html,$(NOTEBOOKS))

# ── Help ────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_%-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  lesson-NN            Run scripts for a single lesson (e.g. make lesson-01)"
	@echo "  notebook-NN          Render a single notebook (e.g. make notebook-03)"

.DEFAULT_GOAL := help

# ── Run all ──────────────────────────────────────────────────────────

.PHONY: all scripts notebooks

all: scripts notebooks ## Run all scripts and render all notebooks

# ── Scripts ──────────────────────────────────────────────────────────

scripts: $(SCRIPTS) ## Run all .r scripts
	@for f in $^; do \
		echo "=== $$f ==="; \
		rustlab run "$$f" || true; \
	done

# Run a single lesson:  make lesson-01  or  make lesson-06
lesson-%:
	@for f in lessons/$*-*/*.r; do \
		echo "=== $$f ==="; \
		rustlab run "$$f" || true; \
	done

# ── Notebooks ────────────────────────────────────────────────────────

notebooks: $(NOTEBOOK_HTMLS) ## Render all notebooks to HTML

notebooks/outputs/%.html: notebooks/%.md | notebooks/outputs
	rustlab-notebook render $< -o $@

# Render a single notebook:  make notebook-03
notebook-%: notebooks/outputs/%-*.html
	@true

notebooks/outputs:
	mkdir -p $@

# ── Clean ────────────────────────────────────────────────────────────

.PHONY: clean clean-scripts clean-notebooks

clean: clean-scripts clean-notebooks ## Delete all generated outputs

clean-scripts: ## Delete only script SVG outputs
	rm -rf lessons/*/outputs/*.svg

clean-notebooks: ## Delete only notebook HTML outputs
	rm -rf notebooks/outputs
