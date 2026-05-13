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
| 5 | Full GPT Architecture | Complete |
| 6 | Training | Complete |
| 7 | Tokenization & Evaluation | Complete |
| 8 | Generation & Capstone | Complete |
| 9 | Modern Architectural Variants (post-curriculum extension) | Complete |
| 10 | Full Backprop and Fine-Tuning (post-curriculum extension) | Complete |

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
- [x] One or more `.rlab` scripts under `lessons/<slug>/` with the required header block
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
- [x] Lesson 13 traces data dimensions through one transformer block (MHA → Add&Norm → MLP → Add&Norm) using `print(size(...))` calls and an ASCII flow diagram in the Theory section
- [x] Lesson 14 scripts print the parameter count of each component and a total. The formula reproduces GPT-2 small at 124,402,944 params (paper: 124,439,808 — within 0.03 %) and GPT-2 medium at 354,724,864 (paper: 354,823,168 — within 0.03 %)
- [x] Lesson 14 includes a bar chart of parameter distribution across embedding, blocks, final LN, and LM head

**Acceptance criteria:**
- A student can derive the parameter count formula $n \approx N \cdot 12 d_{\text{model}}^2 + 2 |\mathcal{V}| d_{\text{model}}$ from Lesson 14 alone.
- No new mathematical concepts introduced — only assembly of previously learned pieces.
- Scripts and notebooks reference specific lesson numbers when reusing concepts (Pre-LN convention from L12, MHA from L09, FFN from L11, etc.).

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 14-full-gpt-architecture
- Next action: Start Phase 6 with `notebooks/15-backpropagation.md`.
- Known blockers for Phase 6: same three rustlab gaps logged for Phase 4 (`M([3,1,2])` row gather, `layernorm(M)` matrix overload, vector vs 1×N matrix arithmetic) plus the two new ones below. Backprop will need analytical gradient derivations until autodiff is available.
- New Phase 6 blocker: **No automatic differentiation in rustlab.** Lesson 15 will derive gradients analytically and code them by hand for one transformer block. Lessons 16–18 (AdamW, LR scheduling, training loop) will need every gradient written out manually until `grad()` or a similar API exists. Worth recording as the next major rustlab feature request.

---

## Phase 6 — Training

**Goal:** Teach the full training pipeline: backpropagation through the transformer, the AdamW optimiser, learning rate scheduling, and the complete training loop with monitoring.

**Lessons in this phase:**
- `15-backpropagation`
- `16-adamw-optimizer`
- `17-learning-rate-scheduling`
- `18-training-loop`

**Deliverables:**
- [x] Lesson 15 visualises gradient magnitude per layer (heatmap) to illustrate vanishing/exploding gradients
- [x] Lesson 16 plots optimiser trajectories (SGD vs. Adam vs. AdamW) on a 2D loss surface
- [x] Lesson 17 plots a warmup + cosine decay LR schedule and overlays it on a simulated loss curve
- [x] Lesson 18 plots training and validation loss curves and gradient norm over steps (and a sister `overfit_demo.rlab` showing the train↘ / val↗ signature)

**Acceptance criteria:**
- Lesson 15 derives the chain rule through at least one attention head explicitly.
- Lesson 16 explains the difference between Adam and AdamW (decoupled weight decay).
- Lesson 17 explains *why* warmup prevents early instability.
- Lesson 18 identifies the signs of overfitting (val loss rises while train loss falls) and underfitting.

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 18-training-loop
- Next action: Start Phase 7 with `notebooks/19-byte-pair-encoding.md`.
- Known blockers for Phase 7: None.
- New rustlab gaps recorded in AGENTS.md Rustlab Recommendations during Phase 6:
  1. **Multi-output function definitions.** `function [a, b, c] = name(...)` is documented as `function [out] = name(...)` only. Workaround: pack outputs into a struct `r = struct("a", ..., "b", ..., "c", ...)` and unpack with `a = r.a` (note: `r.a(i, j)` indexing fails — extract to a variable first).
  2. **Vector vs 1×N matrix arithmetic.** Already logged in Phase 4. Recipe consolidated across Phase 6 lessons: scalarise scalar-valued matrices with `sum(...)`, use `M(1)` to extract row 1 of a 1×N matrix as a vector when chaining elementwise ops, and call `softmax(h * W)` directly rather than via `softmax(logits(1))` (the latter mistakenly indexes a vector as a scalar).

---

## Phase 7 — Tokenization & Evaluation

**Goal:** Cover production-grade tokenization (BPE) and the standard evaluation metric for language models (perplexity). These lessons sit between training and generation.

**Lessons in this phase:**
- `19-byte-pair-encoding`
- `20-perplexity-and-evaluation`

**Deliverables:**
- [x] Lesson 19 visualises the BPE merge frequency as a bar chart and token length distribution as a line plot
- [x] Lesson 20 plots perplexity vs. training steps and a train/val loss comparison showing overfitting

**Acceptance criteria:**
- Lesson 19 walks through 3–5 BPE merge steps manually with printed examples.
- Lesson 20 derives perplexity from cross-entropy loss algebraically (references Lesson 03).
- Both lessons are self-contained — they do not require running any previous script to produce outputs.

**Status:** Complete

**Handoff notes:**
- Last completed lesson: 20-perplexity-and-evaluation
- Next action: Phase 8 complete. The curriculum is complete.
- Known blockers: None.
- New rustlab gap recorded in AGENTS.md during Phase 7:
  3. **`&&` and `||` do NOT short-circuit.** Both operands are eagerly evaluated, which breaks the common idiom `if i < L && seq(i+1) == val ...` (the second operand reads out-of-bounds when `i == L`). Workaround: nest the bound check explicitly, with the second condition inside the first's `if` block. Use a `matched = 0/1` flag if you need to combine many conditions cleanly.

---

## Phase 8 — Generation & Capstone

**Goal:** Implement and compare text generation strategies, then tie the entire series together in a single end-to-end lesson that trains a small GPT and generates text.

**Lessons in this phase:**
- `21-sampling-and-generation` (renamed from `21-sampling-strategies` — scope expanded to cover the autoregressive loop, KV cache, and logit-level controls in addition to the four sampling strategies)
- `22-putting-it-all-together`

**Deliverables:**
- [x] Lesson 21 plots probability distributions under greedy, temperature (T=0.5, 1.0, 2.0), top-K (K=3), and top-P (P=0.9) on the same axes
- [x] Lesson 21 implements the autoregressive generation loop and runs it under every strategy from the Lesson 18 trained model
- [x] Lesson 21 implements naive vs. KV-cached single-head attention forward, verifies bit-equivalent output, and quantifies the FLOP saving (cumulative ~5× at T=8, growing linearly with T)
- [x] Lesson 21 covers repetition penalty, banned tokens / logit bias, and stop conditions
- [x] Lesson 22 is the capstone: a single end-to-end script that tokenizes a small corpus, runs a simplified training loop, checkpoints at intervals, and generates sample text at each checkpoint
- [x] Lesson 22 notebook explicitly references every prior lesson (01–21) by number where each component appears

**Acceptance criteria:**
- [x] Lesson 21 explains the diversity/quality tradeoff for each strategy.
- [x] The Lesson 22 capstone script produces visible improvement in generated text quality across checkpoints. Concrete trace: step 0 outputs random gibberish; step 100 outputs `" at sat sat sat sat sat s"` (model collapsed onto `"sat"`); step 300+ outputs `" at the cat the cat …"` (full bigram learned, greedy mode-collapsed); step 800 with temperature sampling recovers `"sat on the mat"`. Train PPL drops $19.5 \to 1.5$, val PPL drops to $1.4$.
- [x] After Lesson 22, a student has seen every component of GPT built up from scratch.

**Status:** Complete.

**Handoff notes:**
- Last completed lesson: 22-putting-it-all-together (notebook + `capstone.rlab`).
- Next action: The curriculum is complete. Possible future work: implement the full transformer block backward path ([[15-backpropagation]]'s derivations applied to lessons 13/14 forward) and retrain the capstone with the full architecture in the gradient loop.
- ~~Renderer caveat~~ The rustlab 0.3.1 math-escape regression is **fixed in rustlab 0.3.2** (May 12, 2026); a fresh `make notebooks` produces clean book/ output across the curriculum. The workaround note in AGENTS.md is marked ✅ resolved.

**Gap-closing sidebars added during Phase 8 (alongside lessons 21 and 22):**
- Lesson 03 — `## Sidebar: Label Smoothing`
- Lesson 13 — `## Sidebar: Dropout` and `## Sidebar: Encoder–Decoder and Cross-Attention`
- Lesson 14 — `## Sidebar: Initialization` and `## Sidebar: Weight Tying (in practice)`
- Lesson 18 — `## Sidebar: Gradient Clipping`
- Lesson 20 — `## Sidebar: Parallel Evaluation with parmap`

These close the gaps surfaced by the nanoGPT / *Attention Is All You Need* coverage audit (May 2026). They are documentation-only — no script changes — so the existing training scripts continue to run unmodified.

---

## Phase 9 — Modern Architectural Variants (post-curriculum extension)

**Goal:** Cover the four post-2020 architectural deltas that every open LLM (LLaMA, Mistral, Qwen, Falcon) uses against the Vaswani / GPT-2 baseline built in Phases 1–8. Each variant is a surgical swap at one component of the Lesson 14 stack.

**Lessons in this phase:**
- `23-modern-architectural-variants`

**Deliverables:**
- [x] Lesson 23 covers **RoPE** as a rotation matrix replacement for sinusoidal PE (Lesson 10), with a relative-position invariance demonstration.
- [x] Lesson 23 covers **RMSNorm** as a LayerNorm replacement (Lesson 12), with equivalence-on-zero-mean and op-count comparison.
- [x] Lesson 23 covers **SwiGLU** as a GELU-FFN replacement (Lesson 11) with parameter parity at $d_{\text{ff}} = (8/3) d_{\text{model}}$.
- [x] Lesson 23 covers **GQA / MQA** as MHA variants (Lesson 09), extending the KV-cache derivation from Lesson 21 with concrete LLaMA-2-70B-scale memory numbers (20 GB → 2.5 GB).

**Acceptance criteria:**
- [x] Each variant has its own runnable `.rlab` script demonstrating the math.
- [x] Lesson 23 explicitly identifies which earlier lesson each variant swaps against.
- [x] After Lesson 23, a student can read any open LLM source (LLaMA, Mistral, Qwen) and recognise every architectural choice as a delta against the baseline curriculum.

**Status:** Complete.

**Handoff notes:**
- Last completed lesson: 23-modern-architectural-variants (notebook + 4 scripts: `rope`, `rmsnorm`, `swiglu`, `gqa`).
- Follow-on: Phase 10 (Lesson 24) implements the full backward path that Phase 8 deferred, then uses it for SFT and DPO.

---

## Phase 10 — Full Backprop and Fine-Tuning (post-curriculum extension)

**Goal:** Close the "trained end-to-end" gap left by Lesson 22 by implementing the full backward pass through the Lesson 13 single-block transformer, then apply the resulting machinery to two fine-tuning paradigms (SFT and DPO).

**Lessons in this phase:**
- `24-full-backprop-and-fine-tuning`

**Deliverables:**
- [x] Lesson 24 derives the chain rule through one Pre-LN block (LN, softmax row-wise, attention, residuals, FFN-with-GELU, LM head) and wires it into a single backward function.
- [x] Gradient check (numerical finite-difference vs analytical) on `W_U(1, 1)` and `Wq(1, 1)` passes with relative error $\sim 10^{-10}$.
- [x] End-to-end training on the period-3 `abb` corpus drives the loss below the analytic bigram floor (~0.434) to ~$4 \times 10^{-9}$ — concrete evidence the backward pass is correct.
- [x] `sft.rlab` implements supervised fine-tuning with prompt-token loss masking and explicitly probes catastrophic forgetting.
- [x] `dpo.rlab` implements DPO with a frozen reference policy; per-prompt margins +29/+32/+47, post-DPO abb-corpus loss within 10⁻⁵ of pre-DPO (vs SFT's ~2.8 forgetting) — concrete evidence the reference-model term works.

**Acceptance criteria:**
- [x] Each script is self-contained and runs without error.
- [x] Catastrophic forgetting in SFT is demonstrated quantitatively, motivating DPO.
- [x] DPO measurably reduces forgetting relative to vanilla SFT on the same task.

**Status:** Complete.

**Handoff notes:**
- Last completed lesson: 24-full-backprop-and-fine-tuning (notebook + 3 scripts: `full_backprop`, `sft`, `dpo`).
- The full-transformer forward/backward library is duplicated across the 3 scripts (rustlab has no module system; AGENTS.md requires self-contained scripts). When/if rustlab gains imports, factor `forward_only`, `backward_from_dlogits`, `layernorm_fwd/bwd`, and `gelu_grad` into a shared file.
- The Lesson 22 capstone could now optionally be rewritten to train the full architecture instead of the bigram surrogate — currently left as written since the pedagogical point (mode-collapse vs sampling) doesn't change.

---

## Cross-Phase Rules

- **Never skip ahead:** Lessons must be written in order within a phase. Phases may be written in order (1 → 8).
- **No black boxes:** Every mathematical operation in a script must correspond to a named equation in the same lesson's `lesson.md`.
- **Rustlab blockers:** If a needed function is missing, add it to `AGENTS.md` Rustlab Recommendations, implement a workaround, and leave a `# TODO: replace with built-in <function_name> once available` comment in the script.
- **Outputs are not committed:** The `outputs/` directory in each lesson is generated at runtime. Do not add SVG files to git.
- **Status updates:** When finishing a lesson or pausing mid-phase, update the Handoff Notes for that phase in this file.
