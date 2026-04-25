# rustlab_llm

A self-contained tutorial series for building large language models from first principles, using [Rustlab](../rustlab) as the scripting and visualisation environment.

Each lesson pairs step-by-step mathematical theory with runnable Rustlab scripts that produce visualisations. The series builds from raw probability and linear algebra all the way to a complete GPT-style decoder — nothing is a black box.

---

## Goals

- Build genuine intuition for how LLMs work by deriving every algorithm from scratch
- Visualise the mathematics at each step rather than treating components as opaque APIs
- Follow the architecture of [nanoGPT](https://github.com/karpathy/nanoGPT) as the north star
- Assume linear algebra and basic signal processing / information theory as prerequisites — no deep learning background required

---

## Lesson Roadmap

### Phase 1 — Foundations `(Complete)`

| # | Title | Core Concept |
|---|-------|-------------|
| 01 | Tokens & Text Encoding | Characters → integers; vocabulary; one-hot encoding |
| 02 | Probability & Softmax | Probability distributions; softmax; temperature scaling |
| 03 | Cross-Entropy Loss | Comparing predicted vs. true distributions; MLE connection |
| 04 | Embeddings & Similarity | Token → dense vector; cosine similarity; geometry of meaning |

### Phase 2 — First Working Model `(Complete)`

| # | Title | Core Concept |
|---|-------|-------------|
| 05 | The Bigram Language Model | Count-based next-token prediction; train/sample loop |
| 06 | Linear Layers & Gradient Descent | y = Wx + b; loss landscape; gradient step |

### Phase 3 — Attention Mechanism `(Complete)`

| # | Title | Core Concept |
|---|-------|-------------|
| 07 | Context & Naive Averaging | Limitation of single-token context; bag-of-tokens averaging |
| 08 | Scaled Dot-Product Attention | Q, K, V; attention scores; softmax; causal masking |
| 09 | Multi-Head Attention | Parallel attention heads; concatenation & projection |

### Phase 4 — Transformer Components

| # | Title | Core Concept |
|---|-------|-------------|
| 10 | Positional Encoding | Sinusoidal encoding; why attention is order-blind |
| 11 | Feed-Forward Block (MLP) | Position-wise 2-layer MLP; GELU activation |
| 12 | Layer Norm & Residual Connections | LayerNorm stabilises activations; skip connections enable depth |

### Phase 5 — Full GPT Architecture

| # | Title | Core Concept |
|---|-------|-------------|
| 13 | The Transformer Block | Assembling: MHA → Add&Norm → MLP → Add&Norm |
| 14 | Full GPT Architecture | Token embed + pos embed + N blocks + LM head |

### Phase 6 — Training

| # | Title | Core Concept |
|---|-------|-------------|
| 15 | Backpropagation Through Transformers | Chain rule through attention and MLP; gradient flow |
| 16 | AdamW Optimizer | Adaptive learning rates; momentum; decoupled weight decay |
| 17 | Learning Rate Scheduling | Linear warmup + cosine decay; why both matter |
| 18 | The Complete Training Loop | Forward → loss → backward → clip → step → repeat |

### Phase 7 — Tokenization & Evaluation

| # | Title | Core Concept |
|---|-------|-------------|
| 19 | Byte Pair Encoding (BPE) | Iterative merge algorithm; subword vocabulary |
| 20 | Perplexity & Evaluation | Perplexity = exp(loss); train/val split; overfitting diagnosis |

### Phase 8 — Generation & Capstone

| # | Title | Core Concept |
|---|-------|-------------|
| 21 | Sampling Strategies | Greedy; temperature; top-K; top-P (nucleus) |
| 22 | Putting It All Together | End-to-end: tokenize → train small GPT → generate text |

---

## Prerequisites

- Linear algebra (matrix multiply, transpose, eigenvalues)
- Basic probability and information theory (entropy, cross-entropy)
- Some signal processing familiarity is helpful but not required

No prior deep learning experience is assumed.

---

## Running the Lessons

The lessons are integrated **notebooks** under `notebooks/<slug>.md` — prose, math, and code interleaved in a linear narrative. Optional standalone **`.r` scripts** under `lessons/<slug>/` mirror the notebook code blocks for shell-based experimentation.

GitHub renders the executed lessons directly at [`site/`](site/) — that's the landing page. The source notebooks at [`notebooks/`](notebooks/) are what humans edit.

```bash
make                    # show help
make all                # render committed site/<slug>.md + local site/*.html
make notebooks          # regenerate site/<slug>.md from notebooks/<slug>.md
make html               # build site/index.html for local Plotly view (gitignored)
make notebooks-check    # CI drift guard
make lesson-06          # run lesson 06's .r scripts (works for 01–09)
make clean              # delete the interactive HTML build and .r artefacts
```

Or invoke Rustlab directly:

```bash
rustlab run lessons/01-tokens-and-encoding/char_frequencies.r
rustlab                 # interactive REPL
```

Standalone scripts call `savefig("foo.svg")` next to themselves (gitignored). Each `.r` script is self-contained — you can run any single script without running previous ones first.

---

## Repository Layout

This repo follows the [Rustlab lesson-site pattern](../rustlab/docs/lesson-site-pattern.md) shared with [rustlab_em](../rustlab_em/) — sources flat in `notebooks/`, rendered output committed to a top-level `site/`.

```
notebooks/
  README.md            # editor-facing notes (skipped by renderer)
  NN-topic-slug.md     # source notebooks (prose + math + ```rustlab``` blocks)
lessons/
  README.md            # explains the .r-script convention
  NN-topic-slug/
    *.r                # standalone rustlab scripts paralleling the notebook code blocks
site/
  README.md            # hand-written GitHub landing page
  NN-topic-slug.md     # rendered notebook with inline SVG plots (committed)
  plots/NN-topic-slug/ # captured figures (committed)
  index.html           # auto-generated local entry page (gitignored)
  *.html               # interactive Plotly per-notebook (gitignored)
Makefile               # notebooks / html / lesson-NN / clean
PLAN.md                # phase status and handoff notes
AGENTS.md              # project conventions, Rustlab language reference
```

---

## Design Principles

- **No black boxes.** Every mathematical operation in a script has a corresponding named equation in `lesson.md`.
- **Sequential build.** Each lesson depends only on prior lessons; no forward references.
- **Verify by hand.** Scripts print key numerical results so students can check them against pencil-and-paper calculations.
- **One concept per script.** Scripts are kept short enough to read in one sitting.
