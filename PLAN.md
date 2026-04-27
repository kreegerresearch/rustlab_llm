# Project Plan — rustlab_llm

A phased build plan for the nanoGPT tutorial series. Each phase is independently deliverable. Any agent picking up work should:

1. Read this file to understand current status and phase goals.
2. Read `AGENTS.md` for project conventions, Rustlab language reference, lesson format rules, and script conventions.
3. Check the phase's **Status** and **Handoff Notes** section before starting.
4. Update this file's status fields when work is complete or paused.

---

## Overall Status

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Foundations | Complete |
| 2 | First Working Model | Complete |
| 3 | Attention Mechanism | Complete |
| 4 | Transformer Components | Complete |
| 5 | Full GPT Architecture | Not started |
| 6 | Training | Not started |
| 7 | Tokenization & Evaluation | Not started |
| 8 | Generation & Capstone | Not started |

---

## Phase 1 — Foundations

**Goal:** Build the mathematical intuition that every later phase depends on. No neural network code yet — pure math, visualisation, and notation.

**Lessons in this phase:**
- `01-tokens-and-encoding`
- `02-probability-and-softmax`
- `03-cross-entropy-loss`
- `04-embeddings-and-similarity`

**Deliverables — each lesson must have:**
- [x] `notebooks/<slug>.md` with the conventional sections (Learning Objectives, Background, Theory, Standalone Scripts, Expected Numerical Outputs, Exercises, What's next)
- [x] One or more `.r` scripts under `lessons/<slug>/` with the required header block
- [x] All scripts run without error via `rustlab run <path>`
- [x] Each script calls `savefig("foo.svg")` next to itself (artefacts gitignored)
- [x] Key numerical results printed with `print()` so a student can verify by hand

**Acceptance criteria:**
- A student with linear algebra background can read each lesson sequentially with no gaps.
- Lessons 01–04 do not reference attention, GPT, or concepts from Phase 2 onward.
- All scripts are self-contained (no shared state).

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 04-embeddings-and-similarity
- Next action: Start Phase 2 with `lessons/05-bigram-language-model/lesson.md`.
- Known blockers: None.

---

## Phase 2 — First Working Model

**Goal:** Build and train a character-level bigram language model. Students should have a working language model generating text before any attention mechanism is introduced.

**Lessons in this phase:**
- `05-bigram-language-model`
- `06-linear-layers-and-gradient-descent`

**Deliverables:**
- [x] Lesson 05 visualises a bigram frequency matrix and demonstrates the train → sample loop conceptually
- [x] Lesson 06 visualises a 2D loss landscape (heatmap + rotatable `surf`) and shows gradient descent converging
- [x] Both lessons reference Lesson 04 (embeddings) as prior knowledge — no new prerequisites beyond Phase 1

**Acceptance criteria:**
- Students see generated text (even nonsensical) from a trained bigram model after Lesson 05.
- Lesson 06 makes clear why a learned linear layer outperforms a count table.
- Scripts produce at least one heatmap and one line plot per lesson.

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 06-linear-layers-and-gradient-descent
- Next action: Start Phase 3 with `lessons/07-context-and-naive-averaging/lesson.md`.
- Known blockers: None — `for`/`while` loops are fully supported. Gradient descent uses a `for` loop (200 steps). The 2D loss landscape uses the analytic expansion with outer products for efficiency.

---

## Phase 3 — Attention Mechanism

**Goal:** Derive and visualise self-attention from first principles. By the end, students understand scaled dot-product attention and multi-head attention as matrix operations.

**Lessons in this phase:**
- `07-context-and-naive-averaging`
- `08-scaled-dot-product-attention`
- `09-multi-head-attention`

**Deliverables:**
- [x] Lesson 07 shows the failure mode of context-free models and naive bag-of-tokens averaging
- [x] Lesson 08 shows a full attention computation: scores → softmax → weighted sum, with causal mask overlay
- [x] Lesson 09 shows 4+ attention heads side-by-side and explains what concatenation does
- [x] All attention weight visualisations use `imagesc(M, "viridis")` followed by `savefig(...)`

**Acceptance criteria:**
- Attention is derived step by step in Lesson 08 Theory section; Q/K/V dimensions are explicit.
- The causal mask is explained as preventing information leakage from future tokens.
- Multi-head attention motivation (different heads capture different patterns) is stated and illustrated.

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 09-multi-head-attention
- Next action: Start Phase 4 with `lessons/10-positional-encoding/lesson.md`.
- Known blockers: None — matrix slicing (`V1(t,k) = X(t,k)`), `cos`/`sin` on scalars, `pi` constant, and horizontal concat `[A, B]` all confirmed working. Rustlab `function ... end` definitions support matrix returns and compose cleanly (see `causal_attention_weights` helper in lessons 09's scripts).

---

## Phase 4 — Transformer Components

**Goal:** Build the three supporting components that surround attention in a transformer block: positional encoding, the MLP feed-forward sublayer, and layer normalisation with residual connections.

**Lessons in this phase:**
- `10-positional-encoding`
- `11-feed-forward-block`
- `12-layer-norm-and-residuals`

**Deliverables:**
- [x] Lesson 10 plots the full sinusoidal PE matrix as a heatmap and a dot-product similarity vs. distance line plot
- [x] Lesson 11 plots GELU vs. ReLU on the same axes and shows pre/post-activation histograms
- [x] Lesson 12 plots activation distributions before and after LayerNorm and illustrates residual magnitude

**Acceptance criteria:**
- Lesson 10 explains *why* attention without positional encoding is permutation-invariant — proven numerically with a permutation-matrix demo.
- Lesson 11 states why GELU is preferred over ReLU in modern transformers (smooth gradient near zero) — derivative at $x = -0.5$ printed for both.
- Lesson 12 explains vanishing gradients and why residual connections solve them — forward-magnitude collapse vs. preservation across 24 random sublayers.

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 12-layer-norm-and-residuals
- Next action: Start Phase 5 with `notebooks/13-transformer-block.md`.
- Known blockers: None for Phase 5 — `softmax`, `gelu`, `layernorm` (per-vector), `randn`, `seed`, matrix arithmetic all in place.
- New rustlab gaps recorded in AGENTS.md Rustlab Recommendations during Phase 4:
  1. `M([3, 1, 2])` row-vector indexing (workaround: permutation matrix).
  2. `layernorm(M)` matrix overload (workaround: per-row loop with `M(t) = layernorm(M(t))`).
  3. Vector vs. 1×N matrix type distinction in arithmetic — `(W*x')'` returns matrix, breaks `x + ...`. Workaround: use `x * W'` to keep the result a vector. **Worth resolving before Phase 5/6, since the transformer block (Lesson 13) and training loop (Lesson 18) will hit this on every step.**

---

## Phase 5 — Full GPT Architecture

**Goal:** Assemble all components from Phases 3 and 4 into a complete GPT decoder. Students should be able to trace the exact shape of tensors from input tokens to output logits.

**Lessons in this phase:**
- `13-transformer-block`
- `14-full-gpt-architecture`

**Deliverables:**
- [ ] Lesson 13 traces data dimensions through one transformer block (MHA → Add&Norm → MLP → Add&Norm) using print statements and a flow diagram as comments
- [ ] Lesson 14 scripts print the parameter count of each component and a total, matching a known small GPT config (e.g. 6 layers, 6 heads, d_model=384)
- [ ] Lesson 14 includes a bar chart of parameter distribution across embedding, attention, MLP, and output layers

**Acceptance criteria:**
- A student can derive the parameter count formula for a GPT with N layers, H heads, d_model dimensions from Lesson 14 alone.
- No new mathematical concepts introduced — only assembly of previously learned pieces.
- Scripts reference specific lesson numbers when reusing concepts (e.g. `# Softmax: see Lesson 02`).

**Status:** Not started

**Handoff notes:**
- Last completed lesson: —
- Next action: Start with `lessons/13-transformer-block/lesson.md`.
- Known blockers: None.

---

## Phase 6 — Training

**Goal:** Teach the full training pipeline: backpropagation through the transformer, the AdamW optimiser, learning rate scheduling, and the complete training loop with monitoring.

**Lessons in this phase:**
- `15-backpropagation`
- `16-adamw-optimizer`
- `17-learning-rate-scheduling`
- `18-training-loop`

**Deliverables:**
- [ ] Lesson 15 visualises gradient magnitude per layer (heatmap) to illustrate vanishing/exploding gradients
- [ ] Lesson 16 plots optimiser trajectories (SGD vs. Adam vs. AdamW) on a 2D loss surface
- [ ] Lesson 17 plots a warmup + cosine decay LR schedule and overlays it on a simulated loss curve
- [ ] Lesson 18 plots training and validation loss curves and gradient norm over steps

**Acceptance criteria:**
- Lesson 15 derives the chain rule through at least one attention head explicitly.
- Lesson 16 explains the difference between Adam and AdamW (decoupled weight decay).
- Lesson 17 explains *why* warmup prevents early instability.
- Lesson 18 identifies the signs of overfitting (val loss rises while train loss falls) and underfitting.

**Status:** Not started

**Handoff notes:**
- Last completed lesson: —
- Next action: Start with `lessons/15-backpropagation/lesson.md`.
- Known blockers: None.

---

## Phase 7 — Tokenization & Evaluation

**Goal:** Cover production-grade tokenization (BPE) and the standard evaluation metric for language models (perplexity). These lessons sit between training and generation.

**Lessons in this phase:**
- `19-byte-pair-encoding`
- `20-perplexity-and-evaluation`

**Deliverables:**
- [ ] Lesson 19 visualises the BPE merge frequency as a bar chart and token length distribution as a line plot
- [ ] Lesson 20 plots perplexity vs. training steps and a train/val loss comparison showing overfitting

**Acceptance criteria:**
- Lesson 19 walks through 3–5 BPE merge steps manually with printed examples.
- Lesson 20 derives perplexity from cross-entropy loss algebraically (references Lesson 03).
- Both lessons are self-contained — they do not require running any previous script to produce outputs.

**Status:** Not started

**Handoff notes:**
- Last completed lesson: —
- Next action: Start with `lessons/19-byte-pair-encoding/lesson.md`.
- Known blockers: None — `sort(v)` now confirmed available.

---

## Phase 8 — Generation & Capstone

**Goal:** Implement and compare text generation strategies, then tie the entire series together in a single end-to-end lesson that trains a small GPT and generates text.

**Lessons in this phase:**
- `21-sampling-strategies`
- `22-putting-it-all-together`

**Deliverables:**
- [ ] Lesson 21 plots probability distributions under greedy, temperature (T=0.5, 1.0, 2.0), top-K (K=10, 50), and top-P (P=0.9) on the same axes
- [ ] Lesson 22 is the capstone: a single end-to-end script that tokenizes a small corpus, runs a simplified training loop, checkpoints at intervals, and generates sample text at each checkpoint
- [ ] Lesson 22 `lesson.md` explicitly references every prior lesson by number where each component appears

**Acceptance criteria:**
- Lesson 21 explains the diversity/quality tradeoff for each strategy.
- The Lesson 22 capstone script produces visible improvement in generated text quality across checkpoints.
- After Lesson 22, a student has seen every component of GPT built up from scratch.

**Status:** Not started

**Handoff notes:**
- Last completed lesson: —
- Next action: Start with `lessons/21-sampling-strategies/lesson.md` after Phases 1–7 are complete.
- Known blockers: None — all required functions confirmed available.

---

## Cross-Phase Rules

- **Never skip ahead:** Lessons must be written in order within a phase. Phases may be written in order (1 → 8).
- **No black boxes:** Every mathematical operation in a script must correspond to a named equation in the same lesson's `lesson.md`.
- **Rustlab blockers:** If a needed function is missing, add it to `AGENTS.md` Rustlab Recommendations, implement a workaround, and leave a `# TODO: replace with built-in <function_name> once available` comment in the script.
- **Outputs are not committed:** The `outputs/` directory in each lesson is generated at runtime. Do not add SVG files to git.
- **Status updates:** When finishing a lesson or pausing mid-phase, update the Handoff Notes for that phase in this file.
