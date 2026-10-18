# rustlab_llm — render lesson sources to a top-level book/ for GitHub display.
#
#   notebooks/<slug>.md                   editable source notebook (committed)
#   lessons/<slug>/*.rlab                 standalone shell scripts (committed)
#   lib/*.rlab                            shared code pulled in with `run "../lib/<name>.rlab"` (committed)
#   tutorial/sherlock/*.rlab              standalone Sherlock GPT tutorial (train / generate / sft / ask / distill; presets/)
#   book/<slug>.md                        rendered for GitHub (committed)
#   book/plots/<slug>/plot-N.svg          captured figures (committed)
#   book/index.html, book/<slug>.html     interactive Plotly build (gitignored)
#
# Run `make help` for the target list.

BOOK := book
SHERLOCK := tutorial/sherlock
PRESET ?= default
Q ?= Who is the woman?

.DEFAULT_GOAL := help
.PHONY: help all notebooks notebooks-check html validate clean \
	sherlock sherlock-quick sherlock-doctor sherlock-data sherlock-train sherlock-generate \
	sherlock-sft sherlock-distill sherlock-ask sherlock-status sherlock-report sherlock-preset \
	sherlock-bench sherlock-test sherlock-clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo "  lesson-NN          Run .rlab scripts for one lesson (e.g. make lesson-01; 01–26)"

all: notebooks html ## Regenerate the rendered book/ and the interactive HTML build

notebooks: ## Render book/<slug>.md from notebooks/<slug>.md
	rustlab-notebook render notebooks --format markdown --output $(abspath $(BOOK)) --jail-root .

html: ## Build interactive HTML at book/index.html (auto-generated entry page + per-notebook html)
	rustlab-notebook render notebooks --format html --output $(abspath $(BOOK)) --title "rustlab_llm" --jail-root .

notebooks-check: notebooks ## Fail if book/ drifted from sources
	@if [ -n "$$(git status --porcelain -- $(BOOK)/)" ]; then \
		echo "book/ drifted from sources. Run 'make notebooks' and commit." >&2; \
		git status --short -- $(BOOK)/ >&2; exit 1; \
	fi

validate: ## Lint rendered notebooks against markdownlint-cli2 (install: npm i -g markdownlint-cli2)
	rustlab-notebook validate -f markdown notebooks

lesson-%:
	@for f in lessons/$*-*/*.rlab; do echo "=== $$f ==="; rustlab run "$$f" --plot none || exit 1; done

# --- Standalone Sherlock tutorial (tutorial/sherlock/, see its README.md) ---
# Every target runs inside tutorial/sherlock so the scripts' relative paths
# (config.rlab, data/, checkpoints/, ../../lib/gpt.rlab) resolve.  Each stage
# tees its output to <stage>.log; `make sherlock-report` turns the logs and
# figures into report.md, `make sherlock-status` shows how far a run is.
SHELL := /bin/bash
define sherlock_stage
	cd $(SHERLOCK) && set -o pipefail && rustlab run $(1).rlab --plot none 2>&1 | tee $(1).log
endef

sherlock: sherlock-doctor sherlock-data sherlock-train sherlock-generate sherlock-sft sherlock-distill sherlock-report ## Sherlock: run every stage with the selected preset (default: ~30 min), then write report.md
	@echo; echo "all stages done -> $(SHERLOCK)/report.md   (ask it something: make sherlock-ask Q=\"Who is the woman?\")"

sherlock-quick: ## Sherlock: everything with the 3-minute preset (small model, 400 steps, checkpoints-quick/)
	@$(MAKE) --no-print-directory sherlock-preset PRESET=quick
	@$(MAKE) --no-print-directory sherlock

sherlock-doctor: ## Sherlock: check rustlab >= 0.3.9, python3, the corpus and the prepared data
	@command -v rustlab >/dev/null || { echo "rustlab is not on PATH (https://github.com/kreegerresearch/rustlab)"; exit 1; }
	@v=$$(rustlab --version | awk '{print $$2}'); printf 'rustlab %s ' "$$v"; \
	  [ "$$(printf '%s\n0.3.9\n' "$$v" | sort -V | head -1)" = "0.3.9" ] && echo "(ok, >= 0.3.9)" || { echo "(too old: need >= 0.3.9)"; exit 1; }
	@command -v python3 >/dev/null && echo "python3 ok (used once, by prepare_corpus.py)" || { echo "python3 is missing"; exit 1; }
	@test -f $(SHERLOCK)/data/adventures.txt && echo "corpus ok: $(SHERLOCK)/data/adventures.txt" || { echo "corpus missing: $(SHERLOCK)/data/adventures.txt"; exit 1; }
	@test -f $(SHERLOCK)/data/ids.csv && echo "prepared data ok" || echo "prepared data missing (make sherlock-data writes it)"
	@echo "preset: $$(head -1 $(SHERLOCK)/preset.rlab | sed 's/^# *//')"
	@echo "cores: $$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)   (parmap runs the batch across them)"

sherlock-data: ## Sherlock step 0: python3 prepare_corpus.py -> data/ids.csv + data/vocab.rlab (about a second)
	cd $(SHERLOCK) && python3 prepare_corpus.py data/adventures.txt

sherlock-train: ## Sherlock step 1: pre-train the character GPT (config.rlab + preset; ~20 min at the defaults) -> train.log
	$(call sherlock_stage,train)

sherlock-generate: ## Sherlock step 2: three samples from the latest checkpoint (greedy / tau+top-k / tau 1.2) -> generate.log
	$(call sherlock_stage,generate)

sherlock-sft: ## Sherlock step 3: supervised fine-tuning on qa_pairs.rlab, prompt masking, forgetting probe -> sft.log
	$(call sherlock_stage,sft)

sherlock-distill: ## Sherlock step 4: distil the trained teacher into a half-width, half-depth student -> distill.log
	$(call sherlock_stage,distill)

sherlock-ask: ## Sherlock: ask the fine-tuned model: make sherlock-ask Q="Who is the woman?" (no double quotes inside Q) -> ask.log
	@printf 'C.question = "%s";\n' "$(Q)" > $(SHERLOCK)/question.rlab
	cd $(SHERLOCK) && set -o pipefail && rustlab run ask.rlab --plot none 2>&1 | tee -a ask.log

sherlock-status: ## Sherlock: progress report of the running or last run (latest log lines, checkpoints, preset)
	@cd $(SHERLOCK) && echo "preset: $$(head -1 preset.rlab | sed 's/^# *//')"; \
	  for f in train sft distill; do if [ -f $$f.log ]; then echo "--- $$f.log ---"; \
	    grep -E '^\[|held-out|accuracy|forgetting|^done|^(distilled|hard|teacher)' $$f.log | tail -4; fi; done; \
	  echo "--- checkpoints ---"; ls -lt checkpoints*/*.npz 2>/dev/null | awk '{print $$6,$$7,$$8,$$9}' | head -5; \
	  pgrep -fl 'rustlab run' >/dev/null && echo "--- a rustlab run is in progress ---" || true

sherlock-report: ## Sherlock: assemble tutorial/sherlock/report.md from the stage logs and figures
	cd $(SHERLOCK) && python3 report.py

sherlock-preset: ## Sherlock: select a preset for every stage: make sherlock-preset PRESET=quick|large|default (copies presets/<name>.rlab to preset.rlab)
	@test -f $(SHERLOCK)/presets/$(PRESET).rlab || { echo "no preset '$(PRESET)'; available: $$(ls $(SHERLOCK)/presets | sed 's/\.rlab//' | tr '\n' ' ')" >&2; exit 1; }
	cp $(SHERLOCK)/presets/$(PRESET).rlab $(SHERLOCK)/preset.rlab
	@echo "preset '$(PRESET)' selected in $(SHERLOCK)/preset.rlab"

sherlock-bench: ## Sherlock: forward / backward / parmap throughput at several model sizes -> benchmark.log
	$(call sherlock_stage,benchmark)

sherlock-test: ## Sherlock: self-test lib/gpt.rlab (flatten round trip, gradient check, timing)
	cd $(SHERLOCK) && rustlab run test_gpt_lib.rlab --plot none

sherlock-clean: ## Sherlock: delete checkpoints, figures, logs and report.md; reset preset and question (keeps data/)
	rm -rf $(SHERLOCK)/checkpoints $(SHERLOCK)/checkpoints-quick
	rm -f $(SHERLOCK)/*.svg $(SHERLOCK)/*.png $(SHERLOCK)/*.html $(SHERLOCK)/*.gif $(SHERLOCK)/*.log $(SHERLOCK)/report.md
	cp $(SHERLOCK)/presets/default.rlab $(SHERLOCK)/preset.rlab
	@printf 'C.question = "Who is the woman?";\n' > $(SHERLOCK)/question.rlab

clean: ## Delete the interactive HTML build and .rlab script artefacts
	rm -f $(BOOK)/*.html
	rm -f lessons/*/*.svg lessons/*/*.html lessons/*/*.png lessons/*/*.gif
	rm -f notebooks/*.svg notebooks/*.html notebooks/*.png notebooks/*.gif
	rm -rf $(BOOK)/plots/*/.cache
	rm -f $(BOOK)/plots/*/anim-[0-9].gif $(BOOK)/plots/*/anim-[0-9][0-9].gif
