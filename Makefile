# rustlab_llm — render lesson sources to a top-level site/ for GitHub display.
#
#   notebooks/<slug>.md                   editable source notebook (committed)
#   lessons/<slug>/*.r                    standalone shell scripts (committed)
#   site/<slug>.md                        rendered for GitHub (committed)
#   site/plots/<slug>/plot-N.svg          captured figures (committed)
#   site/index.html, site/<slug>.html     interactive Plotly build (gitignored)
#
# Run `make help` for the target list.

SITE := site

.DEFAULT_GOAL := help
.PHONY: help all notebooks notebooks-check html clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo "  lesson-NN          Run .r scripts for one lesson (e.g. make lesson-01)"

all: notebooks html ## Regenerate the rendered site/ and the interactive HTML build

notebooks: ## Render site/<slug>.md from notebooks/<slug>.md
	rustlab notebook render notebooks --format markdown --output $(abspath $(SITE))

html: ## Build interactive HTML at site/index.html (auto-generated entry page + per-notebook html)
	rustlab notebook render notebooks --format html --output $(abspath $(SITE)) --title "rustlab_llm"

notebooks-check: notebooks ## Fail if site/ drifted from sources
	@if [ -n "$$(git status --porcelain -- $(SITE)/)" ]; then \
		echo "site/ drifted from sources. Run 'make notebooks' and commit." >&2; \
		git status --short -- $(SITE)/ >&2; exit 1; \
	fi

lesson-%:
	@for f in lessons/$*-*/*.r; do echo "=== $$f ==="; rustlab run "$$f" || true; done

clean: ## Delete the interactive HTML build and .r script artefacts
	rm -f $(SITE)/*.html
	rm -f lessons/*/*.svg lessons/*/*.html lessons/*/*.png
