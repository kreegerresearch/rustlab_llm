# rustlab_llm — run lessons, render notebooks, clean generated files

SCRIPTS := $(sort $(wildcard lessons/*/*.r))
NOTEBOOKS := $(sort $(wildcard notebooks/*.md))
NOTEBOOK_HTMLS := $(patsubst notebooks/%.md,notebooks/outputs/%.html,$(NOTEBOOKS))

# ── Run all ──────────────────────────────────────────────────────────

.PHONY: all scripts notebooks

all: scripts notebooks

# ── Scripts ──────────────────────────────────────────────────────────

scripts: $(SCRIPTS)
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

notebooks: $(NOTEBOOK_HTMLS)

notebooks/outputs/%.html: notebooks/%.md | notebooks/outputs
	rustlab-notebook render $< -o $@

notebooks/outputs:
	mkdir -p $@

# ── Clean ────────────────────────────────────────────────────────────

.PHONY: clean clean-scripts clean-notebooks

clean: clean-scripts clean-notebooks

clean-scripts:
	rm -rf lessons/*/outputs/*.svg

clean-notebooks:
	rm -rf notebooks/outputs
