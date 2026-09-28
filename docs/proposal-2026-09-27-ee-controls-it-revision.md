# Proposal — rustlab_llm revision: designing an LLM through EE, Modern Controls, and Information Theory

**Date:** 2026-09-27 · **Status:** for review, no code modified · **Baseline:** `main` @ 7e41635, rustlab 0.3.7 (binary built 2026-09-27)

This document answers the three requests in order — (1) `.r` → `.rlab`, (2) what rustlab 0.3.7 now offers, (3) a lesson-by-lesson review with a plan to re-centre the course on the three engineering lenses — and ends with an execution plan and the decisions needed before any code changes.

---

## 0. Recommendations

**Findings in one paragraph.** The `.r` → `.rlab` conversion was completed in May; nothing remains to convert. rustlab 0.3.7 (installed today) closes six of the curriculum's documented workarounds and adds exactly the tools this revision needs — a shared-library mechanism, offline mermaid diagrams, GIF animations, signed heatmaps, multi-panel heatmap grids, integer and fixed-point types, and a controls/DSP toolbox — and a full re-render under it is byte-identical to the committed book. The curriculum itself is complete and numerically accurate on the nanoGPT axis (every Expected-Outputs table matches the book), but it is not yet the course you described: information theory appears in 20 lessons as prose that is rarely computed, and the signals-and-systems and controls vocabulary is essentially absent from all 24 lessons — even where the lessons' own figures already show the physics (the slow eigen-mode in 06, the period-2 pole in 05, the underdamped oscillation in 16, Adam's limit cycle in 18). Four reviewers also found four substantive content errors, one experiment that tests the wrong hypothesis, eight demonstrations whose numbers do not support their prose, three notebooks with zero figures in the rendered book, and a capstone that depends on a lesson two places after it.

**What I recommend, in priority order:**

1. **Do the work in place; do not restart.** The build order is the right design sequence and the mathematics is sound. Re-centre each lesson with a fixed closing section, `## Engineering Lenses` (`### Signals`, `### Systems`, `### Information`), where every statement carries an exact / model / analogy label and at least one computation or figure. This replaces the current prose-only `## Connection to Information Theory`. (§4)
2. **Fix the ordering once, now, before any rewrite:** rename lessons so the general backward pass precedes the capstone — `22-full-backprop-through-the-block`, `23-putting-it-all-together`, `24-modern-architectural-variants`, `25-fine-tuning-sft-and-dpo` — and add `00-the-llm-as-a-system` (course map, notation table, the three lenses) and `26-quantization-and-fixed-point-inference` (new, buildable only since 0.3.7). This touches slugs and cross-links in a mechanical pass; everything after it references stable numbers. (§4.4, §8)
3. **Start with the toolchain pass (Phase 11)** — shared `lib/`, `run` includes, `--jail-root`, remove every 0.3.6 workaround, recombine split figures into subplot grids, make the capstone and fine-tuning lessons execute in-notebook (each script runs in under a second, so the book finally shows their figures), prune orphan SVGs, update AGENTS.md. It is mechanical, low-risk, and every later phase edits the same files. (§2, §6)
4. **Then rewrite the attention/transformer arc (07–14) as one story**: causal time-varying FIR → kernel smoother / correlation receiver with measured row entropies → filter bank → phasor bank → static nonlinearity and keyed memory → forward-Euler state update with a Jacobian check → one system step with both mixing axes → the whole causal system with shape-annotated diagrams and compute budgets. Fix the demonstrations that do not demonstrate (08's near-uniform "full pipeline", 09's permutation W_O, 10's mis-described similarity curve, 12's endpoint-only Pre/Post-LN demo). (§5)
5. **Then the training arc (06, 15–18) as one feedback-control narrative**: derive the stability bound once in 06 and reuse it; draw one closed-loop diagram and fill it in lesson by lesson; explain the oscillations and sawtooth the figures already show; fix 03's gradient section, 16's bias-correction row, 17's experiment. (§3.5)
6. **Then the information arc (01–05, 19–21)**: compute what is asserted — max-entropy softmax, stationary distribution and entropy rate, BPE as source coding with a bits-vs-merges curve, perplexity as arithmetic-coding length, output entropy under each sampling control; repair 20's H(q)/H(p,q) conflation and 21's tied-logit temperature demo. (§3.6)
7. **Finish with the new lessons and a consistency pass** (00, 26, capstone as a design walkthrough with the ablation that settles its headline claim, 25's DPO derivation and the likelihood-displacement finding, notation table, README rewritten for the EE/controls/IT audience). (§6)
8. **Do not add** exercise solutions via the `<!-- solution -->` directive until upstream fixes markdown output; use hand-written `<details>` blocks only where a solution genuinely teaches (derivations), otherwise keep plain numbered exercises. Do not use `tf`/`bode`/`step` for training dynamics — those are continuous-time; the honest analyses are discrete-time with `eig`/`roots`/`expm`, which is also what the reader will reuse. (§7)

Seven decisions are listed in §8 with my recommended answer for each. Approving them as recommended lets Phase 11 start immediately.

---

## 1. Request 1 — convert `.r` to `.rlab`

Already complete. `git log --diff-filter=R` shows the migration commit `83180c2 Migrate script extension from .r to .rlab` on 2026-05-02; `find . -name '*.r'` returns nothing here and in `../rustlab_controls` and `../rustlab_em`. `AGENTS.md`, `README.md`, `Makefile`, `.gitattributes`, and `lessons/README.md` all already refer to `.rlab`.

The only residual is cosmetic: `.gitattributes` still maps `*.rlab` to MATLAB highlighting on GitHub because no rustlab Linguist grammar exists. Nothing to do in this repo.

---

## 2. Request 2 — rustlab 0.3.7: what changed and what it unlocks

The curriculum was last verified against rustlab 0.3.6 (July 2026 build). The installed binary is now 0.3.7, built today from `../rustlab` `main` at `edd2d64`. Everything below was **verified by running it**, not read from the changelog.

### 2.1 Documented workarounds that 0.3.7 makes obsolete

| Curriculum workaround (AGENTS.md → Rustlab Recommendations) | 0.3.7 status | Verified how | Where it lives today |
|---|---|---|---|
| `cache` reserved word → variable renamed `acts` | Fixed (soft keyword) | `cache = 5; print(cache)` runs | 4 `# TODO: rename back to cache` markers in `lessons/22`, `lessons/24` |
| `heatmap`/`imagesc` colour by absolute value → data shifted non-negative | Fixed (signed) | PNG of `imagesc([-2,-1;1,3])` shows −2 darkest, 3 brightest, colorbar −2…3 | 5 notebooks (04, 08, 10, 13, 14) carry `+1e9` / `(PE+1)/2` / `− min` shifts with explanatory comments |
| `subplot` + heatmap SVG export dropped panels 2..n → one figure per block | Fixed | 1×2 subplot heatmap SVG contains both panels | 6 `TODO: recombine into a subplot grid` markers (07, 08×2, 09×2, 13) plus the same split in 11 scripts |
| Bare `figure()` / `histogram()` echoed handle → mandatory `figure();` | Fixed (statement-position `nargout = 0`) | — | Rule can stay (harmless) but no longer needs its warning paragraph |
| `s = svd(W)` bound `U` | Fixed | `svd([3,0;0,1])` → `[3, 1]` | Listed as an open request |
| `rustlab run` exited 0 on runtime error | Fixed (exit 1) | tested | `make lesson-NN` can now gate on failure; `agent-guide.md` §2.1 is stale on this point |
| No warnings for NaN/Inf plot data, unknown colours; `"gray"` ignored by `hline` | Fixed | `hline(2, "gray", ...)` renders | Listed as open request |
| **No module / import system** → transformer library duplicated verbatim in 4 scripts | **Resolved by `run`** | `run "../lib/x.rlab"` works from a script and from a notebook block (see §2.3) | `gelu_grad`, `layernorm_fwd`, `layernorm_bwd`, `backward` ×4 copies; `forward`, `next_dist` ×2 |

### 2.2 Still open after 0.3.7 (keep in AGENTS.md)

- **Autodiff** — none. Analytical backward passes stay.
- **Struct-field indexing** `s.M(2, :)` — still `undefined function 'M'`.
- **`A^k` on a square matrix is element-wise** (rustlab roadmap A8, verified: `[0,1;-1,-0.5]^2 ≠ A*A`). Any controls demo must write `A*A` or `expm`.
- **`<!-- solution -->` emits an unclosed and duplicated `<details>` in markdown output** (roadmap A9, verified). Exercises must stay plain numbered lists, or use hand-written `<details><summary>` HTML, until fixed.
- **Markdown render captures one plot per code block** — a block with three `figure()` calls emits one SVG (`<!-- grid: N -->` only affects HTML). Side-by-side panels in the GitHub book must use `subplot`, which now works for heatmaps.
- New minor findings to file upstream: unquoted `run ../x.rlab` fails to lex (`invalid number: ..`; the quoted form works); `semilogy(y)` requires an explicit x; `freqz` requires three arguments; categorical `heatmap` row labels sit on the cell edge rather than the centre.
- Script-level defects the reviewers found while re-running everything: `perplexity_curve.rlab` passes `"dashed"` in `hline`'s colour slot (0.3.7 now warns); `entropy.rlab` leaves `hold("on")` open; lesson 18's commented-out clipping code calls an untested `sum(M, "all")`; `bigram_sampling.rlab` still mentions `bigram_counts.r` in a comment.

*Lesson numbers throughout this document are the current ones; §8 decision 2 renumbers 22–25.*

### 2.3 New capabilities worth using

| Capability | Verified | Curriculum use |
|---|---|---|
| ```` ```mermaid ```` fences, rendered offline; passed through verbatim to GitHub markdown | yes | Signal-flow / block diagrams for the transformer block (13), full GPT (14), training loop (18), generation loop with feedback (21), DPO wiring (24), course map (00). Replaces the ASCII art in 13/14. |
| `frame()` + `saveanim("x.gif", fps)`; GIF auto-captured into the markdown book | yes | Softmax sharpening vs temperature (02/21), attention rows sharpening as score scale grows (08), optimizer trajectories (16), GD stability sweep (06), residual-stream drift across depth (13). |
| Shared code: `run "../../lib/transformer.rlab"` from scripts; `run "../lib/transformer.rlab"` from notebooks when the Makefile passes `--jail-root .`; `![[_file]]` transclusion of `_`-prefixed markdown (not rendered as its own page) | all three | One `lib/transformer.rlab` (forward, backward, LN fwd/bwd, GELU grad, causal mask, sampling helpers) used by 18, 21, 22, 24 scripts and notebooks. |
| Signed heatmaps; multi-panel heatmap subplots | yes | Remove all shifts; recombine the split figures into 1×3 / 2×2 grids. |
| Column slicing `Q(:, a:b)` and region writes `M(:, a:b) = X` | yes | Replace the 12 nested per-element head-slice copy loops in 13/14 with one-line slices. |
| Complex numbers native: `exp(1j*θ)`, `abs`, `angle`, `polar` | yes | PE pairs and RoPE as phasors: translation = multiplication by `exp(1j*ω*δ)`. |
| Controls/linear algebra: `eig`, `expm`, `lyap`, `tf`, `pole`, `ss`, `ctrb`, `bode`+`savefig`, `step`+`savefig` | yes | GD stability, momentum as a second-order system, adjoint framing (§4). Discrete-time analyses are done directly with `eig` (no `c2d` yet). |
| DSP: `fft`/`fftfreq` (length-preserving), `freqz(b, n, fs)`, `convolve`, `window`, `snr` | yes | Spectrum of PE columns; EMA/momentum frequency response; quantisation SNR. |
| Integer types `int8…uint64` (saturating), `qfmt`/`quantize`/`qadd`/`qmul`/`snr` | yes | New Lesson 25. |
| `tic`/`toc`; `cache enable` persistent function cache; `rustlab-notebook check` linter (currently: 24 files clean) | yes | Timing tables for KV-cache and GQA; CI lint target. |
| `rustlab-notebook render -f pdf` (Catppuccin Latte, needs Inkscape + TeX) | not run | Optional printable course handout. |

### 2.4 Re-render under 0.3.7

`rustlab-notebook render notebooks -f markdown` into a scratch directory took 0.4 s and produced markdown **identical** to `book/` for all 24 lessons. One figure differs: lesson 04 `plot-3` (the king/queen parallelogram) now has the correct legend label `femininity` instead of `scatter` (0.3.7 labelled-scatter fix). Seven orphaned SVGs from older renders sit in `book/plots/` (03, 04, 06×2, 11, 12, 21) and should be pruned.

The heavy scripts are not heavy: `capstone.rlab` 0.85 s, `full_backprop.rlab` 0.23 s, `sft.rlab` 0.27 s, `dpo.rlab` 0.29 s, `train_loop.rlab` 0.32 s. **Lessons 22 and 24 currently have 1 and 0 executable blocks and therefore no figures or captured output in the book.** There is no runtime reason for that; both can execute in-notebook via the shared library.

---

## 3. Request 3 — curriculum review

### 3.1 Method

Four parallel read-only reviews (lessons 01–06, 07–12, 13–18, 19–24), each reading every notebook source in full, the rendered book output, and the scripts, against (a) a canonical concept checklist per lesson, (b) an ease-of-understanding rubric for the target reader, and (c) the three lenses. Every Expected-Outputs table was re-checked against the book; every numerical claim a reviewer doubted was re-run under rustlab 0.3.7; figures were rasterised and inspected where prose describes them. I read lessons 07–14 myself in addition. Full reports (≈ 25 KB each) are in the session scratchpad and can be committed under `docs/review-2026-09-27/` if wanted.

### 3.2 Overall verdict

- **Accuracy is high.** All Expected-Outputs tables match the book. Across 24 lessons the reviewers found four substantive content errors (03 gradient section; 11 mutual-information claim; 12 collapse mechanism and BatchNorm argument; 16 bias-correction row and MAP wording) plus one experiment that tests the wrong hypothesis (17) and a handful of prose-contradicts-figure cases (06, 10, 18).
- **The structure works.** Build order is right; 07→08→09 already builds one coherent mixing-matrix model; 15's derivations are carefully finite-difference-checked; 14 reproduces GPT-2's parameter count to the bias.
- **It is not yet an EE/Controls/IT course.** Information theory appears in 20 lessons but as prose; almost nothing is computed. Signals-and-systems and control vocabulary is absent: no FIR/IIR, impulse or frequency response, state, pole, damping, AGC, phasor, matched filter, adjoint, aliasing anywhere in 24 lessons (one mention of eigenvalues in 16). Where the material itself *exhibits* the engineering story — the slow eigen-mode in 06's unfinished convergence, the period-2 pole in 05's `abab…` sample, the underdamped oscillation in 16's momentum curve, Adam's limit-cycle sawtooth in 18's gradient norm — it sits unexplained on the page.
- **Demonstrations that do not demonstrate** are the most common pedagogical defect: hand-built "interpretable" matrices never interpreted (07, 08, 09), a full-pipeline example whose attention matrix is near-uniform (08), a W_O demo that contradicts its own claim (09), a similarity curve mis-described (10), endpoint-only prints where a curve was promised (12), diagrams referred to that do not exist (12/13), and figures described but not shown (18).
- **Presentation debt** is uniform and mechanical: 6 recombine-TODOs, 5 absolute-value shift hacks, 4 `acts` markers, per-element copy loops, five-fold duplication of the block forward, PMFs drawn as lines, 3×1 stacks where grids belong, autoscaled colourbars, placeholder legends, editor meta-text leaking into student pages (01–04), rustlab-quirk comments inside teaching code (15, 18).

### 3.3 Per-lesson findings (01–18; 19–24 in §3.4)

Severity: **F** fine as is · **L** light polish · **M** moderate rewrite · **X** major rewrite.

| # | Sev | Must fix | Add (lens, label) |
|---|---|---|---|
| 01 | L | Execute the encode/decode round trip (only prose today; 05 then hard-codes its sequence); one corpus throughout; remove editor meta-text; label the space-character bar; show the entropy computation in a visible block | The "staircase of floors": log₂V = 2.807 → H(unigram) = 2.594 → H(X_{t+1}\|X_t) (05) → H(X_{t+1}\|past) (attention) as one bar chart, extended in 05 and 22 (exact). Mermaid text → chars → ids → one-hot → embedding |
| 02 | M | Deliver the stated objective: derive softmax as the max-entropy distribution under an expected-score constraint (six-line Lagrangian); name log-sum-exp and the partition function Z; PMFs as `stem` on 1-based axes, T = 5 shown; rename temperature (τ) to stop the collision with sequence length T | Logits as log-odds, two-class softmax = logistic = soft-decision demapper with LLR (exact); T→0 / T→∞ limits as a `semilogx` entropy-vs-τ curve with `yline` at 0 and log₂V; the softmax Jacobian in one line; GIF of the PMF morphing from argmax to uniform |
| 03 | M | **Replace the gradient section**: it differentiates w.r.t. p̂ and concludes a "100× stronger signal"; the optimiser sees ∂L/∂z = p − y, bounded by 1. Spell out the MLE steps; `contourf` instead of the unreadable `surf`; bits column beside nats; fix the undefined H_unif symbol | p − y as a bounded error signal (exact) — the first appearance of the forward/backward pair, drawn in mermaid; MSE is cross-entropy under a Gaussian likelihood (exact) so 06's MSE is the same objective under a different noise model; compute one KL numerically; state convexity of −z_c + LSE(z) |
| 04 | M | Fix the row/column flip inside one section; signed `heatmap` of E replacing the shifted one with a meaningless colourbar; cut the 16 scalar `cos_sim` calls to three plus the matrix form; label the parallelogram points and use `quiver`; 1-based axes | Dot product as correlator and matched filter, S = ÊÊᵀ as a Gram/correlation matrix, planted here so 08's q·k is "a bank of matched filters" (exact); E as a codebook / VQ (exact); random vectors are nearly orthogonal — histogram of \|cos\| at d = 2, 8, 64 answers "how to choose d" |
| 05 | M | Decode and print the generated text (the book shows a truncated integer vector); one-line row normalisation `C ./ sum(C, 2)` instead of three hard-coded columns; connect row entropies to corpus CE via Σπ_i H(row i) | **Stationary distribution as the eigenvector for λ = 1** (`eig(P')` = {1, 0, −1}, verified): the −1 pole is the period-2 mode that puts `b` on every even position and explains why the empirical unigram ≠ π; smoothing pulls the pole inside the circle; entropy rate = 0.5 bits (all exact). Mermaid state graph; `stem` of π_t under P vs P_smooth. Cite Shannon 1948's own Markov source |
| 06 | X | **Unmet hand-off**: 05 promises "replace the count table with a learned linear layer"; 06 fits y = 2x. **Presented as converged while 5 % of the slow-mode error remains** (H eigenvalues 16.70 / 0.299, κ = 55.8, 0.985²⁰⁰ = 0.049; verified): say so, it is the lesson's most teachable fact. `contourf` with the path overlaid replaces three figures; `semilogy` loss shows the two regimes | GD on a quadratic is the LTI error system e_{k+1} = (I − ηH)e_k: poles 1 − ηλ_i, stable iff η < 2/λ_max (= 0.1198 here — exercise 1 asks for it with no tool), oscillation at η = 0.1, divergence at 0.12 (exact); mermaid feedback loop foreshadowing 16/17; eigenvectors of H with `quiver`; the bigram-as-linear-layer example converging to W_ij = log P_ij + c_i (≈ 25 lines reusing 05's counts) |
| 07 | M | See §5.1 | Causal LTV FIR; EMA as IIR; conditional entropies in bits |
| 08 | M | See §5.2 | Correlation receiver / kernel smoother / associative memory; computed row entropies; GIF |
| 09 | M | See §5.3 | Filter-bank identity; DSP labels; rank; induction-head preview |
| 10 | M | See §5.4 | Phasor bank; aliasing answers "why 10000"; oscillator state update |
| 11 | M | See §5.5 | Wiener–Hammerstein cascade; small-signal gain; keyed memory |
| 12 | M | See §5.6 | Forward Euler; Jacobian product; bus diagram; LN as the guarantor of 08's variance assumption |
| 13 | M | See §5.7 | Two mixing axes; block as one system step; mermaid; shared `block_forward` |
| 14 | L→M | See §5.8; also: the promised "picture inverts at real vocab" is never shown (GPT-2 embedding share ≈ 31 %); "scales to the largest open transformer" overclaims (LLaMA-class breaks the 12d² coefficient — say GPT-2/3 family); `‖H^{(N)}‖ = 70.1` is printed and never discussed (≈ 3× growth over four blocks, motivating the final LN) | FLOPs per token (≈ 2·params forward, 6ND training) and the T where attention FLOPs equal FFN FLOPs; a six-line end-to-end causality test (change `ids(8)`, rows 1–7 of the logits unchanged) — the model as a causal system (exact); logits = ⟨h_f, e_v⟩ under weight tying = a matched-filter bank followed by Gibbs normalisation |
| 15 | M | No computational graph is drawn anywhere; rustlab-quirk comments inside teaching code; "spectral radius" misused for independent random Jacobians (say RMS gain / singular values); the "gradient components are Shannon-like quantities" sentence is wrong; 14 promised LN and GELU backward, neither is derived — derive LN backward or fix the promise; `semilogy` of ‖g_k‖ replaces the 3×12 heatmap | **Backprop is the discrete adjoint / costate recursion** (λ_N = ∂ℓ/∂x_N, λ_k = λ_{k+1} ∂f_k/∂x_k, ∂L/∂θ_k = λ_{k+1} ∂f_k/∂θ_k — Bryson–Ho 1969, LeCun 1988; exact): relabel the existing MLP code with λ and it is unchanged; two-lane mermaid (forward state left→right, adjoint right→left, parameter gradients as rungs); backward through y = xW is the adjoint operator (time reversal for an LTI filter); why reverse mode for a scalar loss; the residual's identity branch as a unity feed-forward path (ρ = 0.7 + I curve inline); p − y as the score function with zero mean at the true model, Adam's v as the diagonal empirical Fisher |
| 16 | M | **The momentum loss curve visibly oscillates (period ≈ 7) and nobody says why** — it is the underdamped heavy-ball pole (\|z\| = √μ = 0.949, verified); unexplained bowl switch a = 200 → 20 → 200; trajectory overlay has no contours; **Expected-Outputs row "bias correction at t = 100 ≈ 1" is false for β₂ (1 − 0.999¹⁰⁰ = 0.095, a 10× rescale)**; "decoupled decay = Gaussian prior" contradicts the lesson's own point; `v` means velocity then second moment; notebook and script demos differ (noise) while the table implies parity | Momentum = one-pole IIR low-pass H(z) = (1−β)/(1−βz⁻¹): pole at β, DC gain 1, time constant 1/(1−β); `freqz` for β ∈ {0.9, 0.99, 0.999} beside the step response; bias correction divides out the step-response deficit (all exact). Heavy ball = second-order system, poles via `roots`, critical damping at ηa = (1−√μ)² (exact). 1/√v̂ = diagonal preconditioning = per-coordinate AGC / normalised LMS (exact); weight decay = leak term (exact). Derive η < 2/λ_max here and reuse in 17. GIF of the three optimisers on contours |
| 17 | M | **The experiment does not test the thesis**: warmup is motivated by Adam's second moment but the demo uses plain SGD with a peak below the stability limit, so a constant η would match it; T_w anchored to 1/(1−β₂) ≈ 1000 while 16 says LLMs use β₂ = 0.95; 76-line block → a `run_sgd(eta_of_t)` helper; "Bayes risk" undefined; the clamped 10⁴ line dominates the loss plot | A schedule makes GD linear time-varying: per-mode gain Π_t \|1 − η_t λ_i\| — plot the contraction factor vs t for the three runs with `hline(1)` (this *is* gain scheduling and replaces the current figure; exact); steady-state variance ∝ ησ²/(2 − ηλ) so decay lowers the noise floor (exact, `loglog`); Adam's step 1 is exactly sign(g) so every coordinate moves by η_max regardless of gradient size — the clean justification for warmup, already printed by `adam_step.rlab`; warmup = soft start (analogy); one sentence on WSD / inverse-sqrt |
| 18 | M | "Reading the diagnostics" describes three plots the reader cannot see; the overfitting signature promised in the objectives is never shown in the notebook; **grad-norm prose ("decays smoothly") contradicts its own sawtooth figure, and exercise 4's premise (peak aligns with LR peak) is false**; 58-line loop with the Adam update written twice → `adamw_step` helper; titled AdamW but λ = 0; clipping code is comment-only; loop is full-batch while the recipe says minibatch | **Training as a closed feedback loop** — one mermaid diagram (plant θ, noisy minibatch sensor, controller = optimiser with integral term, AGC, leak; gain schedule; limiter = clipping; error signal p − y), introduced at the end of 15 and re-shown in 18 with each block filled in as taught (exact framing); clipping as a vector-norm limiter with the knee at c (exact); **the sawtooth is Adam's limit cycle** — near the optimum each coordinate's step stays ≈ η_t and reverses, amplitude ∝ η_t, so log‖g‖ tracks the cosine schedule — plot both on one step axis (exact; this is *why* 17's decay exists); state the val floor 5 ln2/9 = 0.385 explicitly (it equals the observed 0.3851) |

Cross-range notation drift to settle once, in Lesson 00: T = temperature (02) vs sequence length; row vs column conventions (01 rows, 04 flips, 06/11/13 column math over row code, 15 declares rows); bits vs nats with both 2^H and e^L called perplexity; 1-based prose vs 0-based plot axes; gradient symbols ā / dE / g; `v` velocity vs second moment; step index t / k / ℓ; W as mixing matrix (07) vs projection (08+); P as bigram probabilities (05/07) vs permutation (10); X vs H for the token matrix.

### 3.4 Per-lesson findings (19–24)

| # | Sev | Must fix | Add (lens, label) |
|---|---|---|---|
| 19 | M | The "token-length distribution" section promises a histogram and shows three bars over six hand-picked fragments; the IT paragraph is 200 hedged words ending in a claim that is not well-defined on a periodic corpus (entropy rate ≈ 0); `apply_merge` duplicates `bpe_step`; greedy is never justified; byte-level BPE, special tokens (21 later uses EOS, never defined anywhere), and pre-tokenisation are absent; compression ratio and the vocab-size vs sequence-length curve live only in scripts/exercises | **BPE is a compression algorithm** (Gage 1994; identical to Re-Pair) repurposed by Sennrich 2016 — place it relative to Huffman (order-0 optimal) and LZ78 (adaptive dictionary) (exact). Compute total bits for the corpus under 8-bit ASCII (280), fixed ⌈log₂6⌉ (105), the order-0 bound (≈ 78), and after k merges (N_k·⌈log₂\|V_k\|⌉ + merge-table cost); plot bits vs k — the minimum is the MDL-optimal vocabulary and *shows* the diminishing returns now asserted (exact). Bits/char as the tokeniser-invariant bridge to 20. Mermaid merge tree |
| 20 | M | **H(q) vs H(p,q) conflation**: `entropy_to_ppl` and the sweep compute exp H of the model's own distribution while the boxed equation is cross-entropy on data — an IT reader stops here; the 110-line `parmap` sidebar is 35 % of the lesson and about a language feature; the histogram is of an *untrained* model and the prose then explains the interesting shape only appears when trained (training takes 0.1 s); four different numbers are all called "floor"; BPC formula never shown; train/val curve exists only in the script; `hline(3.0, "dashed", …)` passes a style as a colour (0.3.7 now warns) | Build the lesson on **PPL = 2^{H(p,q)} = the antilog of the arithmetic-coding length per symbol** (exact): demonstrate it — Σ −log₂ q(x_t) on the 20-token corpus vs 20·log₂3 = 31.7 bits, ten lines; BPC = (N_tok/N_char)·L/ln 2 with one number; Shannon's ≈ 1 bit/char axis; the per-token loss histogram is the distribution of self-information (mean = CE, std = varentropy → the confidence interval of a PPL estimate, hence long held-out sets) (exact); `semilogy` PPL curve with `yline` floors |
| 21 | M | **The flagship temperature demo is broken**: it runs on P(·\|b) whose two tied logits stay tied at every T (verified: T = 0.5, 1, 2 all give 0.500/0.000/0.500), yet the prose says "T = 0.5 sharper, T = 2.0 nearly random" and the script's figure title repeats it; the KV-cache section prints a **hard-coded string** ("~5e-17") as if computed live; the lesson-18 trainer is pasted verbatim (49 lines); T is temperature and sequence length in the same lesson; beam search absent; lesson 21 says the capstone uses the KV cache and the capstone says it does not | Greedy decoding of a Markov-k model is an autonomous map on a finite state space, so every orbit is eventually periodic — `abab…` is not a failure mode of the bigram, it is the only possibility (exact; 3-node argmax transition diagram); sampling makes it a Markov chain with a Boltzmann tilt; **the generation loop is a closed loop with unit delay and the KV cache is the plant state whose dimension grows with t** (exact; mermaid with the logit controls as a forward-path compensator); top-p as the typical set (exact); GIF of the 10-class distribution as T sweeps with H in the title; `run kv_cache.rlab` to replace the fake print |
| 22 | X | **Zero figures in the rendered notebook** and every number pasted; maintainer changelog in student prose ("the OLD lesson 22…", "22-bigram", the 1.51/√2 note); "computed live in the block above" points below; conclusions precede the objectives; **forward dependency on Lesson 24's backward pass**; "small because rustlab is an interpreter" is no longer true (0.84 s); the headline "attention resolves the ambiguity" is admitted unproven in the same paragraph (PE alone may suffice) | Run the capstone live and plot: `semilogy` loss / grad-norm with the phases named (warm-up transient, exponential convergence, round-off floor — the closed-loop step response of the optimiser); signed heatmaps of A and of W_U Eᵀ; GIF of the attention map across checkpoints; **the three-row ablation** (attention off / PE off / both) that settles the headline claim in 3 s; compute H(next\|cur) = 0.387 nats vs H(next\|cur, prev) = 0 live — attention as the mechanism giving access to I(next; prev \| cur) (exact); MDL reading of PPL 1.00008 (300 params × 64 bits ≫ 133 bits of corpus — a lossless, grossly over-parameterised code; the honest meaning of "memorised"); a design-budget paragraph (why d = 4, 8 merges, 600 steps); mermaid pipeline with lesson numbers |
| 23 | M | Zero figures in the notebook (the RoPE constant-diagonal heatmap and the cache-vs-H_kv curve are script-only); the relative-position identity skips the one step a linear-algebra reader needs (block-diagonal 2×2 rotations commute); "never saw position 9000 … same offsets" contradicts "does not extrapolate zero-shot" three paragraphs later; the "4d+5 vs 2d+3 ops" model is invented; the SwiGLU "on/off switch" is folklore; "any modern open LLM uses all four" contradicts the opening; sliding-window, ALiBi, MoE, FlashAttention, quantisation, speculative decoding appear only as names | **RoPE is complex modulation** (exact, verified to the last digit: `qz .* exp(1i*m*theta)` reproduces `rope_apply`'s dot product): z_k = x_{2k−1} + j x_{2k}, RoPE(x, m)_k = z_k e^{jmθ_k}, score = Re Σ_k q_k k̄_k e^{j(m−n)θ_k} — relative position is the phase difference of two carriers at the same frequency (heterodyne identity); the 13-line loop becomes one line; `polar` spirals for fast and slow pairs. RMSNorm = AGC without the DC block; the shifted-input demo *is* a DC-offset test (exact). SwiGLU = a multiplier between two projections giving the FFN cheap second-order terms (better "why" than a switch). **Decode is bandwidth-bound**: tok/s ≤ HBM bandwidth / cache bytes (20 GB at 3.3 TB/s → 6 tok/s vs 2.5 GB → 50 tok/s) — GQA's operational reason (exact). Sliding-window as a three-line banded mask with heatmap. Quantisation moves to its own lesson (§4.4) |
| 24 | X | **No executable block and zero figures** in the lesson that "opens the black box"; "derive the full backward pass" delivers 2 of ~7 pieces (attention backward, GELU′, CE→logits, W_O, scatter-add are one line or absent; the 100-line `backward` is the only source); **DPO is stated, not derived** (no Bradley–Terry, no KL-regularised objective, no Z cancellation); the SFT "sketch" is pseudo-code in a text fence; the PE confound is admitted but the 0.3 s ablation is left as an exercise; **the script's own output shows likelihood displacement** (after DPO, log-prob of the chosen response is −19.2 and −16.4 nats on two of three prompts while the margin is +29/+32) and the prose says only "the policy prefers chosen over rejected" | The six-line derivation: max_π E_π[r] − β KL(π‖π_ref) ⇒ π* = π_ref e^{r/β}/Z (a Gibbs tilt of the prior with temperature β — Lesson 02's softmax, Jaynes' minimum relative entropy); invert for r; Bradley–Terry cancels Z ⇒ DPO (exact). The DPO weight β(1−σ) as an error-driven gain that vanishes as the margin saturates (dead-band; plot weight vs margin). Forgetting as gradient interference: cos∠(∇L_abb, ∇L_SFT) at the pretrained point; replay and LoRA as "restrict the update subspace"; a rank-1 LoRA demo in ten lines. The finite-difference ε V-curve (`loglog` relative error vs ε) the reader has drawn before. Mermaid DPO wiring with the dashed no-gradient edge into π_ref; pre-train → SFT → DPO/RLHF pipeline. Print lp(chosen) beside the margin and name the pathology (Pal 2024, Razin 2024) |

**Reading order.** Lesson 22 (capstone) trains with the backward pass that Lesson 24 derives; five sentences in 22 point forward to 24. Recommendation in §8.

**Duplication (verified by line diff).** `full_backprop.rlab` shares 186 of 364 non-comment lines with `sft.rlab`, 141 with `dpo.rlab`, 144 with `capstone.rlab`; the three lesson-21 scripts each retrain the lesson-18 model and the notebook does it a fourth time; `bpe_step` exists in 19 and 22; the block forward exists five times across 13/14. A `lib/` with `transformer.rlab` (`layernorm_fwd/bwd`, `gelu_grad`, `forward`, `backward`, `adamw_step`, `causal_mask`, `block_forward`), `bpe.rlab`, `sampling.rlab`, and `bigram_lm.rlab` removes ≈ 700 lines and turns the 22/24 scripts into ≈ 150-line drivers. Packing parameters in a struct removes the 14–17-argument signatures.

### 3.5 Training arc detail (06, 15–18) — the "feedback control" narrative

Today 15→18 reads as four competent recipes; the connective tissue an EE audience expects is never drawn. The plan:

1. **One diagram, filled in progressively.** A mermaid "training as a feedback loop" diagram: error signal (p − y, from 03/15) → adjoint recursion (15) → controller = one-pole filter + per-coordinate AGC + leak (16) → gain schedule (17) → closed loop with a noisy minibatch sensor and a norm limiter (18). Introduced at the end of 15, re-shown in 18 with every block labelled by the lesson that taught it.
2. **Derive the stability bound once (06), reuse it twice (16, 17).** Poles of I − ηH inside the unit circle; the same tool then explains 16's oscillation (heavy-ball poles) and 17's schedule (LTV contraction factor).
3. **Let the figures that already show the physics say so.** 06's slow mode, 16's underdamped oscillation, 18's Adam limit cycle each get one sentence of explanation and one supporting computation.
4. **Fix the experiments.** 17's demo becomes the contraction-factor plot (and, optionally, an Adam run where warmup actually matters); 18 shows the overfitting run it promises, runs clipping live, uses minibatches, and prints how often the limiter fires.
5. **Shared `lib/`**: forward/backward/`adamw_step`/schedule used by 17, 18, 22, 24 so "one step" is visible in a ten-line loop.

### 3.6 Information arc detail (01–05, 19–21)

The IT spine is the course's strongest existing thread; the change is from asserted to computed, plus the two missing exact pieces:

1. **The staircase of floors** (01 → 05 → 22): log₂V, H(unigram), H(X_{t+1}\|X_t), the trained model's cross-entropy, as one recurring bar chart.
2. **Softmax derived, not constructed** (02): max-entropy Lagrangian; Z and LSE named; log-odds; the softmax Jacobian, so 03's p − y is a one-liner.
3. **Bigram as a dynamical system** (05): stationary distribution, spectral gap, entropy rate — the one place the controls lens is *exact* before Lesson 06.
4. **Code length everywhere it is claimed** (03, 19, 20): compute bits/char under chars vs BPE vs the empirical entropy; PPL as effective alphabet size; the LM as a compressor with arithmetic coding as the mechanism.
5. **Output entropy under every sampling control** (21), computed.

---

## 4. The three-lens architecture

### 4.1 Principle

Keep the existing build order (it *is* the design sequence: alphabet → probability → loss → representation → memory-less model → context → attention → block → full model → training → tokenisation → evaluation → inference → variants → fine-tuning). Re-centre every lesson on three recurring questions, asked in a fixed closing section so the reader always knows where the engineering content lives:

- **Signals** — What is the signal (its axis, units, and scale)? What operation is applied to it (filter, transform, modulation, normalisation)? What does it look like in the frequency domain, if that is meaningful?
- **Systems** — What is the state? What is the update law? Is it stable, and what sets the time constant / damping? Where is the feedback?
- **Information** — What bits are created, moved, or destroyed here? What is the floor, the bound, or the budget, and can we *compute* it on the lesson's own data?

Each answer carries one of three honesty labels: **exact** (a formal equivalence), **model** (a faithful simplification), or **analogy** (useful intuition, not an identity). No lens is forced: a lesson that has nothing exact or model-grade to say under a lens says so in one sentence.

Concretely, the existing `## Connection to Information Theory` H2 (present in 20 lessons) becomes `## Engineering Lenses` with up to three H3s — `### Signals`, `### Systems`, `### Information` — and **each H3 contains at least one executed computation or figure**, not only prose.

### 4.2 The cross-cutting map

| LLM component (lesson) | Signals view | Systems view | Information view |
|---|---|---|---|
| Tokens, one-hot (01) | Symbol stream from a discrete source; one-hot = basis / unit impulse (exact) | — | Alphabet size, log₂ V bits per symbol upper bound (exact) |
| Softmax, temperature (02) | Soft-argmax; Boltzmann/Gibbs form with partition function (exact) | — | Softmax is the max-entropy distribution under an expected-score constraint; entropy vs T curve (exact) |
| Cross-entropy (03) | — | Loss as the error signal of the training loop (model) | Code length; CE = H + KL; bits/char (exact) |
| Embeddings (04) | Symbol → vector map (constellation analogy); dot product = correlation / matched filter (exact for the operation) | — | Rank/dimension budget (model) |
| Bigram (05) | — | Markov chain; stochastic matrix; stationary distribution = eigenvector for λ = 1; mixing rate = second eigenvalue (exact) | Conditional entropy floor, entropy rate (exact) |
| Linear layer + GD (06) | — | `x_{k+1} = (I − ηH) x_k`: linear discrete-time system; stable iff η < 2/λ_max; condition number sets the slow mode; eigenvalues on the unit circle (exact) | — |
| Prefix averaging (07) | Causal, time-varying FIR (moving average) along the token axis; EMA exercise = first-order IIR; impulse response = row of W (exact) | — | MI between prior tokens and next token (exact, computed on a toy chain) |
| Attention (08) | Causal FIR whose taps are computed from the signal itself: correlation receiver (q·k) → softmax → weighted sum (exact as an operation; "adaptive filter" is analogy); kernel-smoother / Nadaraya–Watson form (exact); content-addressable memory (exact functional description) | — | Row entropy in bits vs the uniform log₂ t baseline; effect of the 1/√d_k scale on row entropy; rank ≤ d_k of QKᵀ (exact, all computed) |
| Multi-head (09) | Filter bank / subspace decomposition; QK-circuit (routing) and OV-circuit (content) factorisation of each head (exact linear algebra) | — | Joint vs per-head information; redundancy (model) |
| Positional encoding (10) | Bank of oscillators; pair (2k, 2k+1) = phasor e^{jω_k t}; translation = phase rotation e^{jω_k δ}; geometric frequency ladder; fastest wavelength 2π > 2 samples → no aliasing; unambiguous range = slowest wavelength; FFT of PE columns (exact) | — | Order information made recoverable (model) |
| FFN (11) | Memoryless static nonlinearity per sample; Hammerstein/Wiener cascade (analogy); key–value memory (model) | — | Data-processing inequality: no new MI about the next token (exact) |
| LayerNorm + residual (12) | LN = per-sample DC removal + power normalisation, an instantaneous AGC (model) | `x_{l+1} = x_l + f(x_l)`: forward Euler of a continuous flow; Jacobian I + J_f; product of Jacobians and its spectrum across depth (exact) | LN discards 2 of d degrees of freedom; residual sum invertible when f is a contraction (exact, already present) |
| Transformer block (13) | Attention mixes along the time axis, FFN along the channel axis — the two axes of the T×d signal (exact) | One step of a nonlinear discrete-time system on the residual stream; depth = time; magnitude trajectory and Jacobian spectral radius with/without the 1/√(2N) init (exact) | Conditional-entropy refinement (model) |
| Full GPT (14) | Signal-flow graph (mermaid) with shapes on every edge | Whole model as a causal system: input token sequence → output distribution; parameter and FLOP budgets | Bits budget; scaling laws (model) |
| Backprop (15) | — | **Backprop is the discrete-time adjoint / costate recursion** of optimal control (exact); vanishing/exploding gradients = spectral radius of the Jacobian product (exact) | — |
| AdamW (16) | Momentum = first-order IIR low-pass on the gradient with time constant 1/(1−β); `freqz` of the EMA (exact); Adam's second moment = per-coordinate power estimate → normalised step (normalised-LMS / AGC analogy) | Heavy ball = second-order discrete system; characteristic polynomial z² − (1+β−ηλ)z + β; damping and overshoot; decoupled weight decay = leak term (exact) | — |
| LR schedule (17) | Warmup = soft start; cosine decay = annealing (analogy) | Gain scheduling; stability bound η < 2/λ_max revisited with a changing curvature estimate (model) | — |
| Training loop (18) | — | Closed-loop block diagram: plant = model, sensor = loss, controller = optimiser; gradient clipping = saturation / limiter (model) | Train/val loss in bits; overfitting as memorisation (exact) |
| BPE (19) | — | — | Source coding: bits/char under chars vs BPE vs the empirical entropy; vocabulary vs sequence-length trade (exact) |
| Perplexity (20) | — | — | PPL = 2^H = effective alphabet size; LM as compressor (arithmetic coding); Shannon's ~1 bit/char (exact) |
| Sampling + KV cache (21) | Temperature reshapes the spectrum of the output distribution | Generation = closed loop with the model's own output fed back; KV cache = the system state; greedy mode collapse = attractor (model) | Output entropy under each control (exact, computed) |
| Capstone (22) | — | Whole pipeline as one system, executed in-notebook | PPL vs bigram floor (exact) |
| Variants (23) | RoPE = phasor multiplication of q and k so the score depends on phase difference (exact, one line in complex arithmetic); RMSNorm = power-only AGC; GQA = shared taps | — | KV-cache memory budget (exact) |
| Fine-tuning (24) | — | Reference model as a set point; β as a gain (model) | KL-regularised objective; DPO from Bradley–Terry (exact) |
| **Quantisation (new 26)** | Fixed-point arithmetic; quantisation noise; SNR per tensor — measured on N(0, 0.3²) weights with `qfmt`/`quantize`/`snr`: int8 Q1.7 → 36.3 dB, Q3.5 → 24.5 dB, int4 Q2.2 → 6.4 dB, i.e. the 6.02 dB/bit ADC rule the reader knows (exact) | — | Rate–distortion: bits per weight vs perplexity; KL(p‖p_q) of the softmax under quantised logits as the distortion that matters (exact, measured) |

### 4.3 Presentation conventions to enforce course-wide

1. **One notation table** in Lesson 00, reused verbatim: tokens are rows (time axis), features are columns (channels); `X ∈ R^{T×d}`; right-multiplication `XW`; row 0 at the top of every heatmap; queries index rows and keys index columns of every attention matrix.
2. **Every figure gets a one-line "what to look for" callout** (`> [!TIP]`) directly under it, replacing paragraphs that describe a figure the reader cannot see.
3. **Hand-built "interpretable" matrices must be interpreted**: print the resulting matrix and say which entry proves the claim.
4. **Code follows the equation, not the other way round**: replace nested per-element copy loops with slices, factor repeated forward passes into functions from `lib/transformer.rlab`, hide mask/boilerplate construction with `<!-- hide -->`.
5. **Every information-theoretic claim is computed** on the lesson's own data (entropy of a row, MI on a toy chain, bits/char), not only asserted.
6. **Block diagrams in mermaid**, one per architectural lesson, with tensor shapes on the edges.

### 4.4 New lessons and the ordering fix

The capstone (22) trains with the backward pass that 24 derives, and says so in five places. Recommended end sequence:

| New # | Slug | Content |
|---|---|---|
| 00 | `00-the-llm-as-a-system` | **New, small** (prose + 2 figures). The course map as one signal-flow graph of GPT with the three lenses annotated on it; the notation table every lesson reuses; a one-page refresher pointer list (FIR/IIR, discrete-time stability, entropy/KL) so the target reader knows which of their existing tools will be used where. Sorts before `01`. |
| 22 | `22-full-backprop-through-the-block` | The first half of today's 24 (full backward pass, gradient check, end-to-end training on the `abb` corpus), expanded to actually derive the seven pieces it currently asserts. |
| 23 | `23-putting-it-all-together` | Today's 22, executed in-notebook with figures and the ablation. |
| 24 | `24-modern-architectural-variants` | Today's 23, RoPE rebuilt on the phasor form, plus sliding-window and the bandwidth-bound decode argument. |
| 25 | `25-fine-tuning-sft-and-dpo` | The second half of today's 24, with the DPO derivation, LoRA demo, and the likelihood-displacement finding. |
| 26 | `26-quantization-and-fixed-point-inference` | **New, full lesson** (3 scripts). Quantise the Lesson 14/23 weights to int8 with per-tensor and per-channel scales using `qfmt`/`quantize`; weight SNR and perplexity vs bits (6.02 dB/bit measured); KV-cache memory at 16/8/4 bits; the rate–distortion picture; saturation vs wrap; why activations are harder than weights. Genuinely new engineering content that uses rustlab features which did not exist when the curriculum was written. |

Optional, decide after Phase 12: a synthesis lesson **`27-the-transformer-as-a-dynamical-system`** collecting the systems-view material (residual-stream trajectories, Jacobian spectra, Pre-/Post-LN stability, gradient flow) if it grows too large to sit inside 12/13/15. Default: keep it inside 12/13/15.

---

## 5. The attention and transformer arc (Lessons 07–14) — detailed plan

The verdict on this range: **the mathematics is correct and 07→08→09 already builds one coherent "mixing matrix" story (W → A → A_h), but the lessons stop short of the framings this audience needs, several demonstrations do not demonstrate what the prose claims, and 10–12 each lose the thread.** Every lesson in the range is a *moderate rewrite*: keep the structure and the code that works, fix the demos, add one formula and one figure per lens, and delete the 0.3.6 workarounds.

The spine to install and carry through the arc, one formula and one figure per lesson:

> causal time-varying FIR (07) → kernel smoother / correlation receiver with measured row entropy (08) → filter bank with the identity `Concat·W_O = Σ_h A_h X W_V^h W_O^(h)` (09) → phasor bank whose autocorrelation is the similarity curve (10) → linear–static-nonlinear–linear cascade and keyed memory (11) → forward-Euler state update with a Jacobian-product stability check (12) → one step of the system, both mixing axes, signal-flow graph (13) → the whole causal system with shape-annotated graph and budgets (14).

### 5.1 Lesson 07 — Context and Naive Averaging

- **Fix.** The hand-built `X` (rows 5–6 are sums of earlier one-hots) is never read back; the only concrete check (row 3 of X̄ = [⅓,⅓,⅓,0]) lives in the script, not the notebook. Move it in. Fix the IT paragraph: target `X_{t+1} | X_{1..t}` like every later lesson, and the running example (the token that benefits from attending to `mat` is the one after `it`).
- **Signals (exact).** `X̄ = WX` is a causal linear time-varying FIR filter along the token axis; row t of W is the impulse response seen at output time t (a growing box). Figure: `stem` of rows 2, 4, 6 in one `subplot(1,3,·)`; `freqz` of a fixed box vs the EMA of exercise 2 (a first-order IIR, one pole at z = γ) — "smoother" becomes "low-pass", and EMA moves from exercise to body.
- **Information (exact, cheap).** Compute `H(X_{t+1}|X_t)` and `H(X_{t+1}|X_{1..t})` on the two-sentence corpus (1 bit vs 0 bits): the structural failure in bits, priming 08's row entropies.
- **Cleanup.** Recombine X | W | X̄ into a 1×3 subplot; labelled `heatmap` instead of `imagesc`; remove the TODO.

### 5.2 Lesson 08 — Scaled Dot-Product Attention

- **Fix the two demos.** (a) The "interpretable pattern" Q/K is never interpreted: print A and say it — row 2 puts 0.73 on t₁, row 4 puts 0.43 on t₁ vs 0.25 under uniform averaging (content-based retrieval three positions back). (b) The "full pipeline" example with one-hot X and identity-slice projections yields a near-uniform A₂ (row entropies 1.99 vs log₂4 = 2.00 bits, verified); its heatmap is visually the Lesson 07 averaging matrix. Redesign X/W so token 4 retrieves token 1 and the figure shows it.
- **Fix the prose.** The −∞/NaN rationale is wrong for causal masks (the diagonal is always unmasked, so the row max is finite); the honest reasons for −1e9 are dtype portability and fully-masked padding rows. Label the `p(1−p)` gradient remark as a Lesson 15 preview. Say once that Lesson 07's W is now called A and W is now a projection.
- **Signals (exact).** Correlation receiver: `s_{t,i} = q_t·k_i` correlates the template q_t against each stored key; softmax is a soft-argmax detector; 1/√d_k sets the noise floor to unit variance. Kernel-smoother form `o_t = Σ_i K(q_t,k_i) v_i / Σ_i K(q_t,k_i)` with `K = exp(q·k/√d_k)` — Lesson 07 is the box kernel, so 07→08 is one idea. Associative memory: keys = addresses, values = contents, query = probe. Mermaid: `X → {W_Q, W_K, W_V} → S → mask → softmax → A·V → O`.
- **Information (computed, not asserted).** Print `H(A_t)` in bits beside `log₂ t` and `2^H` ("effective tokens attended") for every row; `rank(S) ≤ d_k`. Animation (GIF): one query row's scores and softmax as the scale sweeps 0.1 → 3 with `H(A_t)` in the title — uniform → one-hot in one clip, replacing the two print dumps.
- **Cleanup.** Signed heatmaps: delete the +1e9 shifted mask figure and its apology paragraph; S | masked S | A as a 1×3 subplot; A₂ | O as 1×2; shared `causal_mask(T)` helper from `lib/` (the mask loop is hand-built six times across 08–09).

### 5.3 Lesson 09 — Multi-Head Attention

- **Fix.** Head 2 hand-writes position into Q/K one lesson before Lesson 10 proves attention is order-blind — say so in one sentence and forward-link. The W_O demo uses a permutation, which is exactly the case where W_O mixes nothing, contradicting the claim it illustrates; use a W_O that sums h1.1 and h2.1 into d1 and show it in a signed heatmap. Replace the V1/V2 copy loops with `X(:, 1:2)`; reuse the mask helper.
- **Signals (exact).** The filter-bank identity `Concat·W_O = Σ_h A_h X W_V^h W_O^(h)`: MHA is a sum of H rank-limited time-varying filters with W_O^(h) as synthesis filters — this answers both "why W_O" and "why heads add up". DSP labels for the four heads: head 1 = pick the first sample (DC hold), head 2 = unit delay z⁻¹, head 3 = identity, head 4 = the growing box of Lesson 07 — attention can realise any causal LTV filter and choose it per input. Add the missing argument for several narrow heads: one softmax = one convex combination per position, `rank(Q_h K_hᵀ) ≤ d_k` (print it: 2), H heads = H simultaneous retrievals at the same FLOPs as one wide head. Label heads 2–3 as positional and head 1 as content; one paragraph previewing induction heads (absent from the whole course today).
- **Information.** Replace the qualitative paragraph with a table of per-row entropies for the four heads (≈0 / ≈0.5 / ≈0.5 / log₂ t bits).
- **Cleanup.** The four heads as one 2×2 subplot (the single largest visual win in the range); a labelled heatmap of the packed W_Q with head slices to make 4d² obvious; mermaid fan-out/fan-in; fix the script table ("2×2 grid" vs four SVGs).

### 5.4 Lesson 10 — Positional Encoding

- **Fix.** The similarity-curve prose ("decays through several oscillations, then settles", "unit-ish self-similarity") contradicts the plot (16 → 8 → 10.5 → 6 → 9.6, no settling; self-similarity is d/2 = 16). State the closed form `PE_t·PE_{t+k} = Σ_j cos(ω_j k)` and the curve becomes readable. Fix "derived below/above"; soften the extrapolation overclaim; replace "minimum sufficient encoding" with "a redundant multi-resolution code trading bits for linear decodability" (position needs log₂T bits, PE spends d reals). The equivariance demo has no mask right after three lessons of causal attention: state precisely that with the mask, row t is a permutation-invariant function of {x₁..x_t}.
- **Signals (exact).** Each pair is a phasor `e^{jω_k t}`; translation is multiplication by `e^{jω_k δ}` — with native complex numbers the rotation section is one complex multiply, and RoPE (Lesson 23) becomes "rotate q and k instead of adding to x". Figures: `polar` of three pairs at t and t+δ; `stem` of the 16 frequencies on a log axis (constant-Q coverage from ~1 token to 2π·10⁴); `fft` of one column showing a single spectral line; `semilogy` of wavelength vs pair index (the geometric ladder is stated but never plotted).
- **Aliasing answers "why 10000" (exact).** A single clock ω is unambiguous for k < 2π/ω: the slowest clock sets the maximum unambiguous range (~63 k tokens), the fastest sets the resolution (~1 token). Computation: drop the slow pairs and show `PE_t ≈ PE_{t+63}`.
- **Systems (exact, one line).** `PE_{t+1} = Φ PE_t` with Φ block-diagonal rotations: position is the time index of an autonomous oscillator bank (`expm` of skew blocks).
- **Also add.** Demonstrate relative-offset decodability (fit a linear map PE_t → PE_{t+3}, show zero residual); the cross terms `(e+p)·(e'+p')` in the scores; add-vs-concatenate in one argument. Return to the attention matrix: show A with and without PE (today only exercise 3).
- **Cleanup.** Signed `imagesc(PE)`; drop the `(PE+1)/2` hack; de-duplicate the "embedding ≈ 0.1 vs unit PE" paragraph.

### 5.5 Lesson 11 — Feed-Forward Block

- **Fix.** Column-convention math (`W₁x`, `W₁ ∈ ℝ^{d_ff×d_model}`) vs row-convention code (`H*W1`, `W1 = randn(d_model, d_ff)`) with no acknowledgement — Lesson 08 established rows. Delete the mutual-information paragraph: for continuous variables `I(post; pre)` is infinite wherever the map is locally injective, and GELU is not injective (minimum near x ≈ −0.75); the correct statement is about the Jacobian. Soften "the entire reason GELU displaced ReLU" (empirical). Histograms need ≥ 1000 draws to show the tail described.
- **Signals (exact structurally).** Linear → static nonlinearity → linear, memoryless along the token axis: a Wiener–Hammerstein cascade; attention is the dynamic (time-mixing) stage, so a block is the classic alternation of LTV mixing and static nonlinearity. Small-signal gain: `GELU′(0) = ½`, → 1 for x ≫ 0, → 0 for x ≪ 0 — plot the derivatives that are currently computed but never shown (this pays off in 12).
- **Keyed memory (exact algebra).** `FFN(x) = Σ_m σ(k_m·x + b_m) v_m`: rows of W₁ are keys, columns of W₂ are values, GELU is a sparse gate, d_ff is the number of slots. Figure: signed heatmap of `hidden_post` (T × d_ff) — which slots fire per token. Say explicitly that the FFN holds ≈ ⅔ of block parameters.
- **Cleanup.** 1×2 subplots (activations | derivatives; the two histograms); mermaid L → N → L; the script still carries the `H − min(H)` workaround.

### 5.6 Lesson 12 — LayerNorm & Residuals

- **Fix.** The collapse mechanism: total ratio 4.0e−8 ≈ 2^−24.6 with per-layer geometric mean 0.49 and `gelu′(0) = 0.5` exactly (verified) — once the signal is small GELU is linear with gain ½ and `randn/√d` is norm-preserving on average, so the stack is a cascade with small-signal gain ½ per stage. Say that, not "clipping". Pre-LN growth is presented as a virtue ("exactly the clean gradient highway"); unbounded growth is a known Pre-LN cost and the reason for the 1/√(2N) init and the final LN. The Pre/Post demo overwrites its magnitude each iteration and prints endpoints only — record the curves. The BatchNorm argument is wrong-headed (BN uses running statistics at inference); the real problems are variable-length/padded sequences, cross-example coupling, and train/inference mismatch.
- **Systems (exact).** `x_{l+1} = x_l + h f(x_l)` is forward Euler with h = 0.1 in the demo; depth is time; the no-residual stack is an autonomous map whose origin attracts with linearised gain ½. Add the h = 1 curve (exercise 3 becomes a figure). Compute the product of Jacobians `Π(I + hJ_l)` vs `ΠJ_l` and plot the cumulative smallest singular value per layer — the backward pass is the same product transposed. Pre-LN as integrator: uncorrelated contributions give ‖x_l‖ ∝ √l; plot Pre vs Post with `yline(√d)`.
- **Signals (model, labelled as such).** LN = DC block + per-vector RMS normaliser with a programmable gain/offset (γ, β); memoryless, so not a control loop. The strongest missing "why normalise": LN is what makes the unit-variance assumption behind Lesson 08's 1/√d_k hold at the attention input. Mention RMSNorm here as the power-only variant (Lesson 23).
- **Add.** The residual-stream bus diagram (mermaid) that Lesson 13 already refers to as "the picture from Lesson 12" — it does not exist. Tie the 0.1 factor to the real 1/√(2N) init.

### 5.7 Lesson 13 — The Transformer Block (my reading; reviewer C may add)

- **Add the missing mental model.** Attention mixes along the time axis (rows), the FFN along the channel axis (columns) — the two axes of the T×d signal. State it, draw it, and print one row-mixing and one column-mixing example.
- **Systems (exact).** One block is one step of a nonlinear discrete-time system on the residual stream: `H_{l+1} = H_l + f_l(H_l)`. Track ‖H‖ and the spectral radius of the block Jacobian across a 2-, 4-, 8-block stack with and without the 1/√(2N) branch scaling (the prose already describes the √n growth; make it a figure).
- **Fix code shape.** Replace the ASCII flow with a mermaid signal-flow graph carrying shapes on every edge; replace the eight nested per-element head-slice loops with `Q(:, c_lo:c_hi)`; drop the "purely for readability" per-row LayerNorm loops in favour of `layernorm(H)`; factor the block into `block_forward(H, P)` from `lib/` so the second block is one call instead of 50 duplicated lines.
- **Soften.** "Capacity to lower cross-entropy grows roughly linearly in depth" is not supportable as stated; keep the conditional-entropy refinement picture as a model, not a claim.
- **Cleanup.** H_in | H_mid | H_out as a 1×3 signed subplot; remove the `− min` shifts and the TODO.

### 5.8 Lesson 14 — Full GPT Architecture (my reading; reviewer C may add)

- **Add.** A mermaid architecture diagram with shapes on every edge (the ASCII line today); a FLOPs-per-token estimate (≈ 2 × parameters forward, 6ND for training) next to the parameter count — the compute budget is as much a design constraint as the parameter budget; the crossover T at which attention FLOPs overtake FFN FLOPs (today exercise 5) as a computed line.
- **Systems.** State the whole model as a causal system mapping a token sequence to a distribution, with the residual stream as its state along depth and the KV cache (Lesson 21) as its state along time.
- **Cleanup.** `block_forward` from `lib/` replaces the hidden 45-line helper; slices instead of copy loops; signed logits heatmap without the `− min` shift.

---

## 6. Execution plan

Each phase is independently deliverable, ends with `make notebooks` + `make notebooks-check` + `rustlab-notebook check notebooks` + every touched script run, and updates `PLAN.md` handoff notes and `AGENTS.md` (per the standing rule).

| Phase | Scope | Deliverables | Size |
|---|---|---|---|
| **11 — Toolchain modernisation and ordering** | Whole repo, no pedagogy changes | Renumber 22–25 per §4.4 and fix every cross-link; `lib/` (`transformer.rlab`, `bpe.rlab`, `sampling.rlab`, `bigram_lm.rlab`) with quoted `run` includes in the 13/14/18/20/21/22–25 scripts and notebooks; Makefile `--jail-root .`; remove all abs-shift, recombine-TODO, and `acts` workarounds; recombine split heatmaps into subplot grids; slices instead of copy loops in 13/14; capstone and fine-tuning lessons execute in-notebook so their figures appear in the book; fix the four script defects in §2.2; delete editor meta-text and maintainer changelog from student pages; prune 7 orphan SVGs; AGENTS.md Rustlab Recommendations moved to "Landed 0.3.7" and the still-open list updated; README prerequisites rewritten for the new audience | 1–2 sessions |
| **12 — Attention & transformer arc** | 07, 08, 09, 10, 11, 12, 13, 14 | Rewrites per §5; mermaid diagrams; computed entropies; phasor PE; residual-stream dynamics; the four broken demonstrations fixed; new/revised figures and scripts; expected-output tables regenerated | 3–4 sessions (largest) |
| **13 — Training as feedback control** | 03 (gradient section), 06, 15, 16, 17, 18 | GD stability eigen-analysis and unit-circle plot; momentum as IIR (`freqz`) and second-order system; Adam as normalised step; adjoint framing of backprop; closed-loop training diagram; clipping as limiter; 17's experiment replaced; 18 shows the overfitting run and runs clipping live | 2 sessions |
| **14 — Information arc** | 01, 02, 04, 05, 19, 20, 21 | Staircase of floors; max-entropy softmax; matched-filter framing of the dot product; stationary distribution and entropy rate; bits-vs-merges curve for BPE; PPL as arithmetic-coding length with BPC; output entropy under sampling controls; 21's temperature demo and fake KV-cache print fixed | 2 sessions |
| **15 — New lessons, late lessons, consistency pass** | 00, 22–26 (new numbering), README, PLAN | Lessons 00 and 26; 22 derives the seven backward pieces; 23 as a design walkthrough with live figures and the ablation; 24 with phasor RoPE, sliding window, bandwidth-bound decode; 25 with the DPO derivation, LoRA demo, likelihood displacement; final notation/cross-link/expected-output sweep | 2–3 sessions |

Suggested order: 11 → 12 → 13 → 14 → 15. Phase 11 first because every later phase edits the same files and should start from clean, workaround-free sources with stable lesson numbers.

---

## 7. Risks and constraints

- **Scope discipline.** The lens material must not double lesson length. Budget: at most one new H2 (`Engineering Lenses`) plus targeted edits per lesson; anything larger becomes an exercise or moves to Lesson 26.
- **Honesty of analogies.** Every lens statement carries its exact/model/analogy label; reviewers check them.
- **Markdown renderer limits.** One captured plot per block; solution directive broken; `A^k` element-wise. All three have workarounds noted in §2.2.
- **Controls toolbox is continuous-time** (`tf`, `bode`, `step`); no `c2d`. All training-dynamics analysis is discrete-time and is done directly with `eig`/`expm`, which is also the more honest choice.
- **Expected-output tables** must be regenerated wherever code changes (the 2026-07-12 audit showed how easily they drift).
- **Book drift guard.** Every phase ends with `make notebooks-check` green on 0.3.7.

---

## 8. Decisions requested (with my recommendation)

1. **Template.** Rename `## Connection to Information Theory` → `## Engineering Lenses` with `### Signals / ### Systems / ### Information` H3s, exact/model/analogy labels, and at least one computation per H3. *Recommend yes.* Alternative: keep the IT sections and add a separate `## Signals & Systems View` (more headings, same content).
2. **Ordering and new lessons.** Renumber 22–25 as in §4.4 so backprop precedes the capstone, add `00` and `26`. *Recommend yes, done first in Phase 11.* Alternative: keep numbers, make the capstone forward-reference 24 explicitly (cheaper, but the reading order stays wrong).
3. **Shared library.** Adopt `lib/*.rlab` with quoted `run` includes and a Makefile `--jail-root .`; amend the "each script must run independently" rule to "independently, given `lib/`". *Recommend yes* — it removes ≈ 700 duplicated lines and the 14–17-argument signatures.
4. **In-notebook training.** Execute the capstone, SFT, and DPO inside their notebooks so the book shows their figures. *Recommend yes* — all three run in under a second; the "interpreter is slow" premise is gone.
5. **Synthesis lesson 27.** Decide after Phase 12. *Recommend defer.*
6. **Exercise solutions.** Hand-written `<details>` blocks only for derivation-type exercises; everything else stays a plain numbered list until the upstream `<!-- solution -->` fix lands. *Recommend this middle path.*
7. **Phase order and the review reports.** 11 → 12 → 13 → 14 → 15, and commit the four reviewer reports under `docs/review-2026-09-27/` as the line-numbered work list for the implementation phases. *Recommend yes to both.*
