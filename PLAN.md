**Measured** (14-core Apple silicon, rustlab 0.3.9, default `config.rlab`): model `d = 128, L = 4, H = 4, d_ff = 512, T = 128` → 811,008 parameters. Full default run: 3,000 steps, 3,072,000 tokens (≈ 5.5 epochs), **1,216 s, 2,527 tok/s average** (2,100–2,900 per 50-step window with `parmap` over a batch of 8; short probes up to 3,600). Loss trajectory in bits/char: step 50 → 5.306; step 100 → 4.475 (the order-0 entropy 4.473 — character frequencies learned, nothing else); step 900 → 3.0; step 1,800 → 2.5; **step 3,000 → 2.235 train / 2.203 held-out**, with held-out still falling (2.395, 2.317, 2.266, 2.240, 2.203 at steps 2,000–3,000) — under-trained, not over-fitted. Gradient norm 0.8–1.2 for most of the run (clip at 1.0 active about half the steps). Checkpoints: `ckpt_latest.npz` 19.5 MB (theta + Adam state), numbered snapshots 6.5 MB each. Twenty smoke steps took 9 s.

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
| 10 | Fine-Tuning — SFT and DPO (post-curriculum extension) | Complete |
| 11 | Toolchain modernisation (rustlab 0.3.7), shared `lib/`, lesson renumbering | Complete |
| 12 | Attention & transformer arc rewrite (07–14) under the three lenses | Complete |
| 13 | Training as feedback control (03, 06, 15–18) | Complete |
| 14 | Information arc (01, 02, 04, 05, 19–21) | Complete |
| 15 | New lessons 00 and 26, late lessons (22–25), consistency pass | Complete |
| 16 | Standalone Sherlock tutorial (`tutorial/sherlock/`, `lib/gpt.rlab`) | Complete — one-command `make sherlock-quick` / `make sherlock`; default run, SFT, distillation and benchmark recorded; rustlab request list in `docs/sherlock-performance-and-rustlab-requests.md` |

**Lesson numbering (since 2026-09-27, Phase 11):** `22-full-backprop-through-the-block` (was the first half of 24), `23-putting-it-all-together` (was 22), `24-modern-architectural-variants` (was 23), `25-fine-tuning-sft-and-dpo` (was the second half of 24). The capstone now comes *after* the lesson that derives the backward pass it runs on. Older handoff notes below keep their original numbers where they narrate history.

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
- `22-full-backprop-through-the-block` (split out of the old Lesson 24 in Phase 11 so the capstone follows the backward pass it uses)
- `23-putting-it-all-together` (was `22-putting-it-all-together` until Phase 11)

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
- `24-modern-architectural-variants` (was `23-…` until Phase 11)

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
- `25-fine-tuning-sft-and-dpo` (the SFT/DPO half of the old `24-full-backprop-and-fine-tuning`; the backward-pass half is now Lesson 22)

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
- The full-transformer forward/backward library is duplicated across all scripts using it (rustlab has no module system; AGENTS.md requires self-contained scripts). When/if rustlab gains imports, factor `forward`, `backward`, `layernorm_fwd/bwd`, and `gelu_grad` into a shared file.
- **Lesson 22 capstone rewritten** to use the full-architecture training from this phase. Old bigram-surrogate version reached PPL = 1.51 with greedy collapsing to `"the cat the cat"`; the new full-transformer version reaches PPL = 1.00008 with greedy reproducing the corpus exactly (`"the cat sat on the mat the cat sat on the mat …"`).  Attention resolves the bigram ambiguity that the old capstone could not.

**Full-curriculum accuracy audit (2026-07-12).** Six parallel review passes covered all 24 lessons (every equation hand-checked, every script run, every Expected-Outputs table diffed against captured book output), followed by per-phase fix passes. Highlights: eight latent `M(scalar)`-as-row bugs fixed (lessons 04, 05, 07, 08, 10, 15 — several had shipped visibly-broken book output, including a NaN bigram probability matrix and a backward pass failing its own finite-difference check); lesson 16's three optimizer demos redesigned so their printed numbers support the prose (SGD-vs-Adam, momentum, AdamW-vs-L2); lesson 17's schedule comparison made honest (steps-to-floor 36 vs 127); lesson 19's `bpe_apply` merge list now matches what `bpe_train` learns; lesson 22's bigram-floor claim corrected (PPL 1.4723, computed live in the notebook's first executable block) and its mechanism narrative rewritten for the actual BPE tokenization; lesson 24's SFT comments no longer contradict the script's own forgetting result; stale expected values fixed across lessons 02, 03, 06, 14, 15, 18, 20. Rustlab-side: `cache` reserved-word break worked around (`acts` rename), multi-panel heatmap grids split (subplot+heatmap SVG bug), all `figure()`/`histogram()` echoes suppressed, signed-data heatmaps shifted non-negative (|value| colormap bug). All new rustlab bugs/requests written up in `docs/rustlab-issues-2026-07-12.md` and AGENTS.md → Rustlab Recommendations.

**rustlab 0.3.4 → 0.3.6 maintenance pass (2026-06-07).** Verified the curriculum against rustlab 0.3.6: all 54 `.rlab` scripts execute without error and a full `make notebooks` re-render produces byte-identical plot SVGs (the 0.3.6 `imagesc` y-axis orientation fix affects the interactive/plotters path, not the notebook SVG-export backend — no heatmaps changed). The only book diff was one trailing blank line stripped per `book/*.md` by the 0.3.6 markdown renderer. AGENTS.md updated: the CLI-banner request is now landed (0.3.6 `rustlab run` banner), and the imagesc/axis change, `rustlab cache`, and `rustlab notebook` → `rustlab-notebook` removal are recorded in the Landed section. **Still upstream-blocked:** no autodiff (`grad`) and no module/import system — so the duplicated transformer-library refactor (above) and any autodiff-based capstone rewrite remain deferred; both are now logged as Open feature requests in AGENTS.md.

---

## Phase 11 — Toolchain Modernisation and Ordering (2026-09-27)

**Goal:** Bring the curriculum onto rustlab 0.3.7, remove every workaround the 0.3.6 era forced, factor the duplicated transformer code into a shared library, fix the reading order, and make the capstone and fine-tuning lessons execute inside their notebooks. No pedagogical rewrites — those are Phases 12–15 in `docs/proposal-2026-09-27-ee-controls-it-revision.md`.

**Deliverables:**
- [x] `lib/transformer.rlab` (`causal_mask`, `sinusoidal_pe`, `mha_block_forward`, `transformer_forward`/`transformer_backward`, `ce_dlogits`, `grad_norm`, `adamw_init`/`adamw_step`, `warmup_cosine`, `next_token_dist`), `lib/sampling.rlab`, `lib/bigram_lm.rlab`; scripts `run "../../lib/<name>.rlab"`, notebooks `run "../lib/<name>.rlab"`; Makefile passes `--jail-root .` to the renderer.
- [x] Every duplicated forward/backward copy (capstone, full_backprop, sft, dpo) and block copy (13/14) replaced by the library; all four heavy scripts verified **byte-identical** in stdout against their pre-refactor output.
- [x] Lessons renumbered 22–25 (see Overall Status); old Lesson 24 split into notebooks 22 and 25; all cross-links, README, Makefile, and this file updated.
- [x] Lessons 22, 23, 25 execute in-notebook (previously 0–1 code blocks and no figures in the book); their Expected-Outputs tables regenerated from the live run.
- [x] All `TODO: recombine into a subplot grid` splits recombined into `subplot` figures; all `|value|`-colormap shifts removed (signed heatmaps); `acts` → `cache` revert; editor meta-text removed from lessons 01–04; maintainer changelog removed from the capstone; script defects fixed (`hline("dashed")`, unclosed `hold`, `.r` comment, `sum(M, "all")`).
- [x] Book re-rendered; orphan plot SVGs pruned; `make notebooks-check` and `rustlab-notebook check` clean; every script exits 0.

**Handoff notes:**
- Renderer facts that shaped this phase (also in AGENTS.md): the markdown renderer captures **one plot per code block**, so side-by-side panels use `subplot`; a `run` inside a script that a *notebook* runs resolves relative to the notebook, so notebooks carry their own short library-based blocks rather than running the lesson scripts; `run` paths must be quoted.
- Next action: Phase 12 (attention & transformer arc, lessons 07–14) per §5 of the proposal, starting from these clean sources.
- Known blockers: none. Still upstream-open: autodiff, struct-field indexing, `A^k` element-wise, `<!-- solution -->` in markdown output.

---

## Phases 12–15 — Engineering Lenses Revision (2026-09-27)

**Goal:** Turn the nanoGPT curriculum into a course on designing an LLM from signals-and-systems, modern control, and information theory, per `docs/proposal-2026-09-27-ee-controls-it-revision.md` (approved as recommended) and the four reviewer reports in `docs/review-2026-09-27/`.

**Deliverables:**
- [x] Every lesson 01–26 closes with `## Engineering Lenses` (`### Signals` / `### Systems` / `### Information`); each present lens has ≥ 1 executed block and every claim opens with **Exact.** / **Model.** / **Analogy.** (AGENTS.md → Engineering Lenses). The former prose-only `## Connection to Information Theory` sections were merged into `### Information` as computations.
- [x] Lesson 00 (`00-the-llm-as-a-system`): course map as a mermaid signal-flow graph with the two feedback loops, the three lenses, the notation table, the toolkit-to-lesson map.
- [x] Lesson 26 (`26-quantization-and-fixed-point-inference`): b-bit weight quantisation of the capstone model, Δ²/12 and 6.02 dB/bit, PPL/KL vs bits, weights vs activations, saturate vs wrap, KV-cache memory and the bandwidth bound; three scripts.
- [x] Phase 12 (07–14): the attention arc as one story — causal LTV FIR (07) → correlation receiver / kernel smoother with measured row entropies and a softmax-sharpening GIF (08) → filter-bank identity and DSP head labels (09) → phasor bank, aliasing answers "why 10000", oscillator state update (10) → Wiener–Hammerstein cascade, small-signal gain, keyed memory (11) → forward Euler, Jacobian products, Pre/Post-LN curves, LN as the guarantor of 08's variance assumption (12) → two mixing axes, block as a system step, mermaid graph (13) → causality test, FLOPs budget, matched-filter head (14). All demonstrations that did not demonstrate were fixed (08's full pipeline, 09's permutation W_O, 10's similarity prose, 12's endpoint-only demo).
- [x] Phase 13 (03, 06, 15–18): p − y replaces the wrong gradient section (03); GD as the LTI error system with poles, η_crit = 0.1198, and the bigram-as-linear-layer example (06); the adjoint/costate theorem, computational graphs, and the closed-loop training diagram (15); momentum as a one-pole IIR, heavy-ball poles, Adam as normalisation, the β₂ bias-correction row fixed (16); the LTV contraction-factor experiment replaces the SGD demo (17); overfitting run shown, clipping live, grad-norm ringing explained with measured mechanism (18).
- [x] Phase 14 (01, 02, 04, 05, 19–21): staircase of floors (01 → 05); softmax derived by max-entropy, τ for temperature, LLR demapper (02); correlator / matched-filter bank and codebook view (04); stationary distribution and the period-2 pole, entropy rate (05); BPE as source coding with the bits-vs-merges curve, byte-level BPE, EOS defined (19); PPL = 2^{H(p,q)} demonstrated as code length, BPC, varentropy (20); tied-logit temperature demo explained honestly, live KV-cache check, closed-loop generation, argmax orbits (21).
- [x] Phase 15 (22–25): the seven backward pieces derived against the library's lines, ε V-curve, ablation table (22); design budget, four-row ablation landing on the bigram floor, attention GIF, MDL reading (23); RoPE as complex modulation verified bit-identical, sliding window, bandwidth-bound decode with honest numbers (24); DPO derivation, gain vs margin, gradient interference, rank-1 LoRA (25).
- [x] Book re-rendered (27 pages, all figures live); `rustlab-notebook check` clean; every script (106) exits 0.

**Handoff notes:**
- Line budgets: most lessons grew 35–60 %; 06, 10, 22, 24 grew more because their plans mandated many new artefacts. Candidates for trimming if wanted are listed in the reviewer hand-backs (e.g. Lesson 10's FFT or polar panel → exercise; Lesson 26's Q-format example).
- Honest corrections made while executing the plan: the proposal's "6 tok/s vs 50 tok/s" decode figures did not follow from their inputs (Lesson 24 now computes the bound properly); "Adam's per-coordinate step ≈ η_t" was not what Lesson 18's data shows (the notebook states the measured mechanism); the capstone's old "previous-token head" reading was false for the trained model (rewritten from printed entries); Lesson 05's row-entropy floor prints as √2 = 1.4142.
- rustlab findings recorded in AGENTS.md (Rustlab Recommendations): `svd` absolute floor ≈ 1e-8; `hline` after a plot replaces the series unless `hold("on")`; the SVG backend drops line/scatter series overlaid on `contour`/`contourf`/`imagesc`; `quiver` needs gridded origins; `V(:, i)'` does not transpose a column slice; indexed `+=` does not parse; integer-matrix `*` is element-wise; `snr`/`histogram` reject 1×n matrices; `print` caps at 16 arguments; `${expr}` directly followed by `	imes$` or `> 1$` is not interpolated; wikilinks inside callouts are not transformed (use plain links); no `vline`.
- `lib/` follow-ups requested by the rewrites (not done, to keep this phase's diff reviewable): `lib/bpe.rlab` (`bpe_step`, `apply_merge`, `tok_str`), `lib/capstone.rlab` (`capstone_tokens`, `train_capstone`), `lib/quant.rlab`, `add_grads`/`grad_dot`/`flat_grad` and costates from `transformer_backward`, `stationary_dist`, `conditional_entropy_bits`, `mi_bits`, `gd_modes`, a 1-based `sinusoidal_pe`, and a `train_bigram_lm` variant returning per-step losses.
- Next action: none scheduled; the curriculum is complete under the approved plan. Optional Lesson 27 (the transformer as a dynamical system) remains deferred per decision 5.

## Phase 16 — Standalone Sherlock tutorial (2026-10-18)

**Goal:** Take the curriculum's components off the toy corpora. A standalone, `make`-driven tutorial under `tutorial/sherlock/` pre-trains a character-level GPT on *The Adventures of Sherlock Holmes* (Project Gutenberg #1661, public domain; 607 KB raw, 562,269 characters after cleaning, 77-symbol vocabulary, order-0 entropy 4.473 bits/char), samples from it, fine-tunes it on hand-written Q/A pairs with prompt masking and a forgetting probe, and distils it into a 2-layer student — with every knob in one `config.rlab`, honest time and quality expectations (word-shaped Sherlock-flavoured text at 2.2 bits/char after the 20-minute default run, real spelling and names only with longer runs or more books, memorised answers after SFT; no reasoning), and a measured account of what rustlab would need to go faster. No notebook: this is the first part of the repo whose scripts are complete programs rather than one-concept demonstrations.

**Deliverables:**
- [x] `lib/gpt.rlab` — trainable decoder-only GPT of arbitrary depth and head count (token embedding + sinusoidal PE + $L$ Pre-LN blocks + final LN + untied LM head); per-layer matrices stacked along rows; flat parameter view (`gpt_flatten` / `gpt_unflatten`) so the optimiser, checkpoints and `parmap` gradient sums are one-liners; `gpt_forward` / `gpt_backward` / `gpt_grad_flat`; `gpt_adamw`, `gpt_lr`, `gpt_clip`; `gpt_generate` (greedy / temperature / top-k); `gpt_encode` / `gpt_decode`. API summary in the file header.
- [x] `tutorial/sherlock/test_gpt_lib.rlab` — flatten round trip, finite-difference gradient check on 12 random coordinates and on every one of the 14 parameter blocks (pass if rel. err. < 1e-6), serial and `parmap` timing at the default size. `make sherlock-test`.
- [x] `tutorial/sherlock/prepare_corpus.py` (Python 3, standard library only — rustlab cannot read text files or map characters to codes) → `data/ids.csv` + `data/vocab.rlab` (gitignored); `data/adventures.txt` committed. Handles multiple input files, `--lower`, `--max-chars`, Gutenberg header/footer stripping; prints the order-0 entropy.
- [x] `config.rlab` (one struct `C`: data, model, optimiser / schedule, logging / checkpoints, generation), `train.rlab` (data-parallel over the batch with `parmap`, AdamW + warmup/cosine + global-norm clip, held-out evaluation on fixed windows, NPZ checkpoints with Adam state, resume, `train_curves.svg`), `generate.rlab` (greedy, $\tau$ + top-k, $\tau = 1.2$ untruncated; reads model dims from the checkpoint's `cfgv`).
- [x] Presets and a question CLI: `config.rlab` ends with `run "preset.rlab"`; `presets/{default,quick,large}.rlab` (empty / 2–3 min, d 64 L 2 T 64 400 steps / overnight, d 192 L 6 H 6 20,000 steps batch 14); `ask.rlab` + `question.rlab` put one question to `ckpt_sft.npz` in the SFT template (`C.use_base_model = true` falls back to the pre-trained model).
- [x] `Makefile` targets `sherlock` / `sherlock-quick` / `-doctor` / `-data` / `-train` / `-generate` / `-sft` / `-ask Q=…` / `-distill` / `-status` / `-report` / `-bench` / `-test` / `-preset PRESET=…` / `-clean`, listed by `make help`; root `README.md` section; `AGENTS.md` layout, Running paragraph and the "tutorial scripts may exceed 60 lines" rule; `lessons/README.md` pointer; `.gitignore` for checkpoints, generated data, figures and logs.
- [x] `tutorial/sherlock/README.md` — what you build and what to expect, prerequisites, quick start, every `config.rlab` field with 5-minute and overnight settings, resuming, what each printed column means and what good values are (6.27 / 4.47 / ≈ 3.3 / ≈ 1.5–1.8 bits/char), reading `train_curves.svg`, generation controls, the SFT and distillation stages, scaling to the full canon, troubleshooting, file map, lesson cross-references.
- [x] `sft.rlab` + `qa_pairs.rlab` — 46 Q/A pairs in a `Q: …\nA: …` template with the mask on the answer characters only; 300 steps of fresh AdamW at a constant 1e-3 with 50 % replay (3e-4 learned too slowly in 300 steps) of corpus windows (`C_sft` struct at the top of `sft.rlab`); greedy-answer probes during training incl. one never-trained paraphrase; final accuracy over all 46 and the held-out-loss forgetting probe before → after; writes `checkpoints/ckpt_sft.npz`, `sft_curves.svg`.
- [x] `distill.rlab` — Hinton-style distillation into a half-width, half-depth student (at the defaults 2 layers, $d = 64$, 4 heads, 109,440 params): loss $\alpha \tau^2\,\mathrm{KL}(p_\tau^{T} \| p_\tau^{S}) + (1-\alpha)\,\mathrm{CE}$ at $\tau = 2$, $\alpha = 0.5$; `kd_dlogits` replaces `gpt_dlogits` and is finite-difference checked; a hard-target control student is trained from the same seed, init and windows; prints teacher / distilled / hard-target held-out bits/char and the distillation gain; writes `checkpoints/ckpt_student.npz`, `distill_curves.svg`. Knobs in the `C_kd` struct at the top of the script.
- [x] `benchmark.rlab` — model-size sweep (d 64/128/192/256) with serial and `parmap(8)` tok/s and GFLOP/s, cost breakdown of the default model by forward/backward part, primitive timings at the tutorial's shapes, `parmap` scaling over the batch, memory per configuration, `benchmark.svg`.
- [x] `docs/sherlock-performance-and-rustlab-requests.md` — the benchmark tables (sweep 109 k → 6.35 M parameters, serial and `parmap` tok/s, achieved GFLOP/s, forward/backward breakdown, `parmap` scaling, memory), where the time goes (65 % matmul at 7.4 GFLOP/s on complex storage, no BLAS linked; 24 % of a step outside the windows in unflatten / reduction / AdamW), what a Sherlock-scale run costs today, and eight ranked rustlab requests with expected gains.
- [x] Ease of use: `make sherlock-quick` / `make sherlock` run every stage and finish with `report.md`; `make sherlock-doctor` (toolchain, data, preset, cores), `make sherlock-status` (progress report of a running or finished run from the stage logs), `make sherlock-report` (`report.py` assembles logs + figures into `report.md`); every stage tees to `<stage>.log`; the training log prints `[percent done  ETA min:sec]` every `log_every` steps; the `quick` preset trains into `checkpoints-quick/`; the distillation student is derived from the teacher (half width, half depth) so presets stay consistent.
- [x] `tutorial/sherlock/example-run/` — the complete default run committed as documentation: `train.log`, `generate.log`, `sft.log`, `distill.log`, `ask.log`, the four figures and the assembled `report.md`.
- [x] A completed default run recorded here (see **Measured** below) and its final numbers and end-of-run sample in the tutorial README.

**Measured** (14-core Apple silicon M4 Max, rustlab 0.3.9, 2026-10-18): default model `d = 128, L = 4, H = 4, d_ff = 512, T = 128` → 811,008 parameters. Pre-training: 3,000 steps × 1,024 tokens = 3.07 M tokens ≈ 5.75 epochs in 1,216 s (2,527 tok/s average; 2,100–2,900 tok/s per log line); loss 5.306 bits/char at step 50 → 4.475 at step 100 (the order-0 entropy, i.e. character frequencies learned) → 2.235 train / **2.203 held-out bits/char** at step 3,000, still falling (under-trained, not over-fitted). Samples: word-shaped Sherlock-flavoured prose with real function words and dialogue punctuation, most content words misspelled; greedy decoding degenerates into a repeated 4-gram after ~150 characters (Lesson 21's failure mode). SFT (46 pairs, 300 steps, batch 8 with 4 replay windows, constant LR 1e-3): 113 s; **43 / 46 = 93.5 %** exact greedy answers ("Who is the woman?" → "Irene Adler", "Where does Holmes live?" → "221B Baker Street", "Who narrates the stories?" → "Dr. Watson"); forgetting probe 2.203 → 2.445 held-out bits/char (+0.243); without replay the same recipe at 3e-4 lost 1.9 bits/char on the corpus, textbook catastrophic forgetting; the never-trained paraphrase is never answered correctly (memorisation, not understanding). Distillation (600 steps, τ = 2, α = 0.5) into the 109,440-parameter student: 82 s + 42 s for the hard-target control; teacher 2.203, distilled **3.420**, hard-target 3.439 bits/char (gain +0.020 bits/char at this scale). Benchmark sweep (serial / `parmap(8)` tok/s): 109 k params 3,925 / 18,435; 811 k 814 / 4,106; 2.69 M 270 / 1,299; 6.35 M 120 / 463; `parmap` speedup 1.7× / 3.0× / 4.5× / 5.2× at B = 2 / 4 / 8 / 14. Quick preset (d 64, L 2, H 4, d_ff 256, T 64; 101 k parameters; 400 steps × 512 tokens): pre-training 18 s at 11,624 tok/s to 3.403 held-out bits/char; SFT (800 steps via `C.sft_steps`, 37 s) 29 / 46 exact answers, forgetting +0.112 bits/char; distillation into a 17,568-parameter student 3.529 vs 3.586 bits/char hard-target; the whole `make sherlock-quick` pipeline (doctor, data, train, generate, SFT, distillation, report) in under five minutes; `sft.rlab` left-truncates the 11 pairs that exceed the 65-token context.

**Handoff notes:**
- **Performance doc.** `docs/sherlock-performance-and-rustlab-requests.md` (with `benchmark.rlab`) is the single place for timings and for the upstream asks; the tutorial README and this file link to it by name rather than repeating numbers that will go stale.
- **rustlab facts that shaped the design** (also in AGENTS.md → Rustlab Recommendations where they are requests): no text-file reader and no char ↔ code conversion (hence the Python step); string literals have no escape sequences (newline = `sprintf("%c", 10)`, which `vocab.rlab` relies on for `"`, `\`, newline and tab); `parmap` lambdas must be pure and receive copies, so the only parallelism is one window per worker with the gradient rows summed afterwards; `A^k` is element-wise; no autodiff (every gradient in `lib/gpt.rlab` is hand-derived and checked). `lib/gpt.rlab` has no KV cache, so generation re-runs the full forward per character.
- **Next steps, in rough order of value:** (1) merge-time check that `make sherlock-quick` still passes under the current rustlab; (2) more books — `prepare_corpus.py a.txt b.txt …`, the full canon ≈ 3.4 MB ≈ 3.2 M characters, ≈ 20 min per epoch at the default model; (3) a BPE tokeniser (Lesson 19) to cut sequence length 3–4× and let `T = 128` cover a paragraph; (4) a KV cache (Lesson 21) for generation; (5) replay during SFT to blunt forgetting, and DPO (Lesson 25) on preference pairs; (6) fixed-point inference of the distilled student (Lesson 26); (7) `tutorial/` scripts for a second corpus to show the pipeline is corpus-agnostic.
- **Rules:** never edit `lib/gpt.rlab` without re-running `make sherlock-test`; changing `d_model` / `n_layer` / `n_head` / `d_ff` or the vocabulary invalidates `checkpoints/` (`train.rlab` trusts `config.rlab`; `generate.rlab` reads `cfgv` from the checkpoint); `tutorial/` scripts keep the script header and section conventions but may exceed 60 lines.

---

## Cross-Phase Rules

- **Never skip ahead:** Lessons must be written in order within a phase. Phases may be written in order (1 → 8).
- **No black boxes:** Every mathematical operation in a script must correspond to a named equation in the same lesson's `lesson.md`.
- **Rustlab blockers:** If a needed function is missing, add it to `AGENTS.md` Rustlab Recommendations, implement a workaround, and leave a `# TODO: replace with built-in <function_name> once available` comment in the script.
- **Shared code:** anything used by more than one lesson goes in `lib/*.rlab` and is pulled in with a quoted `run` — never copy a function between scripts.
- **Outputs are not committed:** The `outputs/` directory in each lesson is generated at runtime. Do not add SVG files to git.
- **Status updates:** When finishing a lesson or pausing mid-phase, update the Handoff Notes for that phase in this file.
