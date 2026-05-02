# rustlab_llm — render lesson sources to a top-level book/ for GitHub display.
#
#   notebooks/<slug>.md                   editable source notebook (committed)
#   lessons/<slug>/*.rlab                 standalone shell scripts (committed)
#   book/<slug>.md                        rendered for GitHub (committed)
#   book/plots/<slug>/plot-N.svg          captured figures (committed)
#   book/index.html, book/<slug>.html     interactive Plotly build (gitignored)
#
# Run `make help` for the target list.

BOOK := book

.DEFAULT_GOAL := help
.PHONY: help all notebooks notebooks-check html clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo "  lesson-NN          Run .rlab scripts for one lesson (e.g. make lesson-01)"

all: notebooks html ## Regenerate the rendered book/ and the interactive HTML build

notebooks: ## Render book/<slug>.md from notebooks/<slug>.md
	rustlab notebook render notebooks --format markdown --output $(abspath $(BOOK))

html: ## Build interactive HTML at book/index.html (auto-generated entry page + per-notebook html)
	rustlab notebook render notebooks --format html --output $(abspath $(BOOK)) --title "rustlab_llm"

notebooks-check: notebooks ## Fail if book/ drifted from sources
	@if [ -n "$$(git status --porcelain -- $(BOOK)/)" ]; then \
		echo "book/ drifted from sources. Run 'make notebooks' and commit." >&2; \
		git status --short -- $(BOOK)/ >&2; exit 1; \
	fi

lesson-%:
	@for f in lessons/$*-*/*.rlab; do echo "=== $$f ==="; rustlab run "$$f" || true; done

clean: ## Delete the interactive HTML build and .rlab script artefacts
	rm -f $(BOOK)/*.html
	rm -f lessons/*/*.svg lessons/*/*.html lessons/*/*.png
