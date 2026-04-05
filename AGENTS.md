# AGENTS.md — Tutorial Template

This file guides AI coding tools working in this repository.

## Where to Start

Before writing any lesson, read `PLAN.md` to find the current phase status and handoff notes. Update `PLAN.md` when completing a lesson or pausing work mid-phase so the next session can pick up cleanly.

---

## Project Purpose

`rustlab_llm` is a self-contained learning project exploring **LLM tutorial using RustLab** through simulation.
Each lesson pairs markdown theory notes (with equations) with runnable rustlab scripts that produce visualizations.
The goal is to build intuition by seeing the math come alive as we learn how to construct and train LLMs and GPT's similar to nanoGPT.   

**Learning goal: To make a series of lessons on the subject of large language models.  The core algoirthms including the attention networks and core algorithms, Traingin and inefernce.  provide a simple example.

**Prerequisite knowledge: Linear algebra, Some Signal Processing and Information Theory

This repo is **independent from rustlab** — never modify `../rustlab` from here. Instead, note needed features in the "Rustlab Recommendations" section below.

---

## Lesson Roadmap

Each lesson builds on previous ones. Scripts visualize and demonstrate the mathematics; theory is in `lesson.md`.

### Phase 1 — Foundations

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 01 | Tokens & Text Encoding | Characters → integers; vocabulary; one-hot encoding | Character frequency bar chart; one-hot matrix heatmap | Planned |
| 02 | Probability & Softmax | Probability distributions; softmax; temperature scaling | Softmax curves at T=0.5, 1.0, 2.0; entropy bar chart | Planned |
| 03 | Cross-Entropy Loss | Comparing predicted vs. true distributions; MLE connection | Cross-entropy surface plot vs. predicted probability | Planned |
| 04 | Embeddings & Similarity | Token → dense vector; cosine similarity; geometry of meaning | Embedding matrix heatmap; cosine similarity matrix | Planned |

### Phase 2 — First Working Model

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 05 | The Bigram Language Model | Count-based next-token prediction; train/sample loop | Bigram frequency heatmap; sample text generation trace | Planned |
| 06 | Linear Layers & Gradient Descent | y = Wx + b; loss landscape; gradient step | 2D loss landscape contour; gradient descent path overlay | Planned |

### Phase 3 — Attention Mechanism

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 07 | Context & Naive Averaging | Limitation of single-token context; bag-of-tokens averaging | Token mixing weight matrix; averaged vector comparison | Planned |
| 08 | Scaled Dot-Product Attention | Q, K, V; attention scores; softmax; causal masking | Attention score matrix heatmap; causal mask overlay | Planned |
| 09 | Multi-Head Attention | Parallel attention heads; concatenation & projection | Side-by-side heatmaps for 4 attention heads | Planned |

### Phase 4 — Transformer Components

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 10 | Positional Encoding | Sinusoidal encoding; why attention is order-blind | Sinusoidal PE heatmap; dot-product similarity vs. distance | Planned |
| 11 | Feed-Forward Block (MLP) | Position-wise 2-layer MLP; GELU activation | GELU vs. ReLU comparison; pre- and post-activation distributions | Planned |
| 12 | Layer Norm & Residual Connections | LayerNorm stabilises activations; skip connections enable depth | Activation distributions before/after LayerNorm; residual magnitude plot | Planned |

### Phase 5 — Full GPT Architecture

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 13 | The Transformer Block | Assembling: MHA → Add&Norm → MLP → Add&Norm | Data-flow dimension trace through one block | Planned |
| 14 | Full GPT Architecture | Token embed + pos embed + N blocks + LM head | Parameter count breakdown bar chart; dimension flow diagram | Planned |

### Phase 6 — Training

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 15 | Backpropagation Through Transformers | Chain rule through attention and MLP; gradient flow | Gradient magnitude per layer heatmap | Planned |
| 16 | AdamW Optimizer | Adaptive learning rates; momentum; decoupled weight decay | Optimizer trajectory comparison (SGD vs. Adam vs. AdamW) | Planned |
| 17 | Learning Rate Scheduling | Linear warmup + cosine decay; why both matter | LR schedule plot; loss curve sensitivity to schedule | Planned |
| 18 | The Complete Training Loop | Forward → loss → backward → clip → step → repeat | Training & validation loss curves; gradient norm over time | Planned |

### Phase 7 — Tokenization & Evaluation

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 19 | Byte Pair Encoding (BPE) | Iterative merge algorithm; subword vocabulary | Merge frequency bar chart; token length distribution | Planned |
| 20 | Perplexity & Evaluation | Perplexity = exp(loss); train/val split; overfitting diagnosis | Perplexity vs. training steps; train vs. val loss comparison | Planned |

### Phase 8 — Generation

| # | Title | Core Concept | Key Rustlab Output | Status |
|---|-------|-------------|-------------------|--------|
| 21 | Sampling Strategies | Greedy; temperature; top-K; top-P (nucleus) | Probability distribution plots for each strategy; diversity comparison | Planned |
| 22 | Putting It All Together | End-to-end: tokenize → train small GPT → generate text | Full training run loss curve; sample generations at checkpoints | Planned |

**Numbering rule:** two-digit with leading zeros. Keep lessons ordered so each one depends only on prior lessons.

---

## Tool: Rustlab

Rustlab is a scientific computing CLI (`../rustlab`) with a scripting language optimized for matrix and signal processing work. Run scripts with:

```bash
rustlab run lessons/01-[topic]/[script_name].r
```

Run `rustlab` (no args) for an interactive REPL. Full reference: `../rustlab/README.md`.

### Language Essentials

- Imaginary unit is `j`; complex literal: `z = 3.0 + j*4.0`
- 1-based indexing: `v(1)` is the first element; `v(end)` is the last
- Suppress output with `;`; comment with `#`
- Element-wise ops: `.^`, `.*`, `./`; matrix multiply: `*`; conjugate transpose: `'`
- Column vector: `v = [a; b; c]`; matrix: `M = [a, b; c, d]`
- Range: `1:10`, `0:0.1:1`, `10:-1:1`

### Standard Function Reference

**Math:** `exp(v)`, `log(v)`, `log2(v)`, `log10(v)`, `sqrt(v)`, `abs(v)`, `sin(v)`, `cos(v)`, `tanh(v)`, `acos(v)`, `asin(v)`, `atan(v)`, `real(v)`, `imag(v)`, `conj(v)`, `angle(v)`

**Stats:** `sum(v)`, `min(v)`, `max(v)`, `mean(v)`, `std(v)`, `cumsum(v)`, `sort(v)`, `argmin(v)`, `argmax(v)`, `trapz(v)`

**ML / Activations:** `softmax(v)`, `relu(v)`, `gelu(v)`, `layernorm(v)`, `layernorm(v, eps)`

**Array:** `zeros(n)`, `ones(n)`, `eye(n)`, `linspace(a,b,n)`, `len(v)`, `length(v)`, `numel(x)`, `size(x)`

**Matrix:** `reshape(A, m, n)`, `repmat(A, m, n)`, `transpose(A)` / `A.'`, `diag(v)` / `diag(M)`, `horzcat(A,B,...)` / `[A B]`, `vertcat(A,B,...)` / `[A; B]`, `rank(M)`, `eig(M)`, `det(M)`, `trace(M)`, `outer(a,b)`, `kron(A,B)`, `expm(M)`

**Random:** `rand(n)` (uniform vector), `randn(n)` (normal vector), `randn(m, n)` (normal matrix), `randi(imax,n)`, `randi([lo,hi],n)`

**I/O:** `save(file, x)`, `save(file,"name",x,...)`, `load(file)`, `load(file,"name")`

**Plotting — file output:**
```r
savefig(v, "outputs/plot.svg", "Title")
savestem(v, "outputs/stem.svg", "Title")
saveimagesc(M, "outputs/heat.svg", "Title", "viridis")  # colormaps: viridis, jet, hot, gray
savebar(x, y, "outputs/bar.svg", "Title")
savescatter(x, y, "outputs/scatter.svg", "Title")
savehist(v, n_bins, "outputs/hist.svg", "Title")
```

**Plotting — multi-series / subplots:**
```r
figure()
subplot(2, 1, 1)
hold("on")
plot(x, y1, "color", "blue", "label", "train")
plot(x, y2, "color", "red",  "label", "val", "style", "dashed")
title("Loss")
xlabel("Step")
ylabel("Loss")
legend()
subplot(2, 1, 2)
plot(grad_norms)
title("Gradient Norm")
savefig("outputs/training.svg")
```

### Topic-Specific Functions

| Function | Why it matters for this tutorial |
|---|---|
| `softmax(v)` | attention weights and output probabilities |
| `gelu(v)` | MLP feed-forward activation (Lessons 11, 13, 22) |
| `layernorm(v)` | normalisation after every attention and MLP block |
| `relu(v)` | ReLU comparison vs. GELU (Lesson 11) |
| `sum(v)` | cross-entropy loss; probability normalisation checks |
| `log(v)` | cross-entropy: `-sum(target .* log(probs))` |
| `mean(v)`, `std(v)` | activation distribution diagnostics (Lesson 12) |
| `cumsum(v)` | top-P nucleus sampling threshold (Lesson 21) |
| `sort(v)` | top-K sampling: sort logits, slice top K, renormalise (Lesson 21) |
| `reshape(A, m, n)` | split/merge attention heads: `[T, d_model]` ↔ `[T, n_heads, d_head]` (Lessons 09, 13) |
| `tanh(v)` | activation function comparison: relu / gelu / tanh on same axes (Lesson 11) |
| `outer(a,b)` | attention score matrix: `scores = outer(q, k)` pattern |
| `cos(v)`, `sin(v)` | sinusoidal positional encoding (Lesson 10) |
| `randn(n)` | weight initialisation with normal distribution |
| `saveimagesc` | attention weight heatmaps; PE matrix (Lessons 08–10) |
| `savebar` | character frequencies (L01), parameter counts (L14), BPE merges (L19) |
| `savescatter` | embedding geometry (L04), optimiser trajectories (L16) |
| `savehist` | activation distributions before/after LayerNorm (L12) |
| `subplot` | side-by-side attention head comparison (L09) |
| `save` / `load` | persist intermediate results for capstone (L22) |

---

## Directory Structure

```
lessons/
  NN-topic-name/
    lesson.md        # theory, equations, learning objectives, simulation guide
    *.r              # rustlab scripts — one script per concept
    outputs/         # generated SVG/PNG — created at runtime, NOT committed
```

Each `.r` file covers exactly one coherent concept so students can run pieces independently.

---

## Lesson Format (`lesson.md`)

Use GitHub-flavored Markdown with LaTeX math: `$inline$` and `$$block$$`.

**Required sections (in order):**

1. `## Learning Objectives` — 3–5 bullet points stating what the student will be able to do
2. `## Background` — prerequisite knowledge assumed for this specific lesson
3. `## Theory` — derivations and key equations; derive step-by-step, name every variable, state units
4. `## Core Concepts` — the central idea expressed plainly in 1–2 paragraphs before any math
5. `## Simulations` — for each `.r` file: what it computes, what to observe in the output, what to verify by hand
6. `## Exercises` — 3–5 follow-up questions or script modifications that deepen understanding

**Style guidance:**
- Derive equations step-by-step; never drop a step without explanation
- Name every variable and state units explicitly
- Connect math to intuition: explain *why* the result looks the way it does
- If a concept has a common misconception, call it out explicitly

---

## Script Conventions (`.r` files)

**Required header block:**
```r
# Script:  [filename].r
# System:  [what physical/mathematical system is being modeled]
# Concept: [the single concept this script demonstrates]
# Equations: [key equations, in plain text or LaTeX-style comments]
# Units:   [what units are used for each physical quantity]
```

**Additional rules:**
- Separate logical sections with blank lines and `# === Section Name ===` comments
- All plot output goes to `outputs/` relative to the script's directory
- Always `print()` the key numerical results a student should verify by hand
- Name files descriptively: `gradient_descent_1d.r`, not `plot1.r`
- Keep scripts short enough to read in one sitting — split if over ~60 lines
- Each script must run independently (no shared state between scripts)

---

## Running Lessons

Run from the project root:

```bash
# Lesson 01 — [Title]
rustlab run lessons/01-[topic]/[script1].r
rustlab run lessons/01-[topic]/[script2].r

# Lesson 02 — [Title]
rustlab run lessons/02-[topic]/[script1].r

# Interactive exploration:
rustlab
```

Generated SVGs appear in `lessons/<NN-topic>/outputs/`. Open them in any browser or SVG viewer.

---

## Rustlab Recommendations

When a concept in this tutorial requires a function or feature that rustlab does not yet have, record it here. Be specific: describe what the function should do, what inputs it takes, what it returns, and why it is needed to express the concept cleanly.

**Format:**
```
### [function_name](args) → return_type
**Needed for:** Lesson NN — [title]
**Purpose:** [what it computes]
**Why rustlab needs it:** [why existing functions don't cover this]
**Example usage:**
  result = function_name(arg1, arg2)
```

<!-- Add recommendations below this line -->

All previously requested functions have been implemented. No outstanding recommendations.
