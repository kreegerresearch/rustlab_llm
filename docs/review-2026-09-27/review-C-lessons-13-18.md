# Review C — Lessons 13–18 (transformer block → training loop)

Scope: `notebooks/13..18-*.md` read in full; `book/` diffs and all 15 `lessons/1[3-8]-*/*.rlab` scripts skimmed; the scripts whose numbers the book does not capture (train_loop, overfit_demo, optimizer_comparison, adam_step, sgd_vs_momentum, param_count, two_block_stack, gradient_flow) were re-run under rustlab 0.3.7 in the scratchpad, and the rendered SVGs were rasterised and inspected. Line numbers refer to the notebook sources.

---

## Lesson 13 — The Transformer Block

**1. Covers now.** Pre-LN equations + ASCII flow (L21–36); shape trace; one block forward with explicit per-row LN and per-head slicing loops; magnitude tracking; two-block stack (code copied verbatim); per-block parameter count; IT section; dropout and encoder–decoder sidebars. Figures: 3 separate `imagesc` panels (H_in/H_mid/H_out minus min). `two_block_stack.rlab` has a bar chart the notebook omits. No real diagram.

**2. Clarity problems (ranked).**
1. Code is ~2× longer than the idea. L90–146 (57 lines) and L238–297 (60 lines) are the same block twice with `2` suffixes; both `.rlab` scripts already factor `block_fwd`. Define it once (or `run` a shared file) and stacking becomes three lines, which is the whole point of the section.
2. Figures do not support the prose. L228 asks the reader to compare three separate panels and "zoom in" to see nudges; each panel is rescaled by its own min, axes are numeric 0–8 with half-integer ticks, no token/feature labels. Show H_in, A_out (MHA delta) and F_out (FFN delta) side by side, signed.
3. Stale 0.3.6 workaround comments at L208–210 (subplot TODO, |value| colormap).
4. Notation: FFN written column-style `W_2 GELU(W_1 x + b_1)` at L168 while all code and Lesson 15 use row convention `x W`. Row-by-row `layernorm` loops "purely for readability" (L91–93) appear four times; `layernorm(M)` is one line.
5. Weak "why": L84 explains Pre-LN, but nothing explains *why two sublayers* — the token-mixing vs feature-mixing split is only implicit at L344.
6. Loose claims: L343 "H orthogonal channels" (heads are not orthogonal); L346 "capacity grows roughly linearly in depth" (unsupported).

**3. Missing / thin.**
- The two axes of the T×d matrix: attention is the *only* operation that mixes rows (tokens), FFN mixes columns (features) per row. Needs to be a headline plus a 6-line demo: perturb token 3's input, show MHA changes rows ≥3 while FFN changes only row 3.
- Controls hook x_{l+1} = x_l + f_l(x_l): absent (Lesson 12 L114 already has "gradient highway", so this can point back).
- Signal-flow block diagram with shapes on arrows: ASCII only.
- L395 "attention dominates compute at long context" asserted, never derived (only Lesson 14 Ex 5).

**4. EE / controls / IT reframing.**
- *Exact:* the Pre-LN block is one forward-Euler step of a discrete-time system on the residual-stream state; the final LN + head is the readout. Figure: mermaid signal-flow with the residual as the state line; plus `semilogy` of ‖x_l‖ vs l for N=8 random blocks with and without the 1/√(2N) output scaling ("open-loop gain per stage").
- *Exact:* a softmax row is a causal, non-negative, unit-DC-gain FIR tap vector that depends on the input — attention is an adaptive causal filter across time, FFN a memoryless nonlinearity per sample (a Wiener/Hammerstein cascade; loose but instructive, and it links back to Lesson 07's uniform-FIR averaging). Computation: `sum(A_h, 2)` = 1, so attention cannot amplify a constant.
- IT section (L339–348) is fine as prose; drop the "linear in depth" claim.

**5. Rustlab 0.3.7.**
- ```mermaid for L30–36 and a second one for the encoder–decoder sidebar (L372).
- `<!-- grid: 3 -->` + signed `imagesc`/`heatmap` with token/feature labels for L207–226; drop `- min`.
- `run block_lib.rlab` (shared `block_fwd`, mask) consumed by L13, L14, L15's attention head, L24.
- `<!-- details: per-head slicing -->` around L115–141 if the loops stay inline.

**6. Defects.** No numerical defects: 6.07/9.20/10.54, 0.987, 0.521, 14.85, 808/840 all match. Expected-Outputs table lacks the magnitude rows the prose leans on (add ‖H_in‖, ‖H_mid‖, ‖H_out‖, ‖H_out2‖). Bias convention is inconsistent-looking (FFN biases counted, attention biases not, LN affines not) — L318/L337 explain it, but state the convention once at the top of the count.

**7. Severity: moderate rewrite** — maths correct, but code is doubled, figures don't deliver the comparison, and both conceptual hooks the owner wants (two mixing axes; residual stream as dynamical system) are absent.

---

## Lesson 14 — Full GPT Architecture

**1. Covers now.** ASCII pipeline (L21–25), equations, component table, toy forward (V=50, T=8, d=64, N=4), logits heatmap, parameter formula + toy bar chart, GPT-2 small reconstruction with the 36,864 bias discrepancy explained, IT prose, initialization and weight-tying sidebars. Figures: 1 heatmap (logits − min), 1 bar chart.

**2. Clarity problems (ranked).**
1. `block_fwd` (L108–157, 50 lines) is fully hidden; a reader who wants to confirm it is Lesson 13's block cannot. `<!-- details -->` or `run`.
2. L272 promises "at a real-vocab config (next example) the picture inverts", but the GPT-2 example prints only a total; the ~31 % embedding share is never shown.
3. Stale colormap workaround at L206–207.
4. Overclaim at L296 and L360 ("scales to the largest open transformer", "handles GPT-3 without modification"): GPT-3 yes; LLaMA-class (SwiGLU with three d×8d/3 matrices, GQA, no biases, RMSNorm) breaks the 12d² coefficient. Say "GPT-2/3 family" and forward-ref Lesson 23.
5. `|H^{(N)}| = 70.1` (L173) is printed and never discussed: 512 unit-variance entries would give 22.6, so the stream grew ~3× over four blocks — this continues Lesson 13's growth story and motivates the final LN (L44).
6. L259 "fits in a 1 MB file": 205,440 × 4 B = 0.82 MB in f32 (1.6 MB in f64) — say f32.
7. 13 lines of nested loops for PE (L69–81); Lesson 10 presumably has this — hide or `run`.

**3. Missing / thin.**
- FLOPs per token: absent (checklist). ≈ 2·n_params forward, ≈ 6·n_params training, plus ≈ 4·T·d per layer for attention; compute for the toy config and GPT-2 and give the T where attention FLOPs equal FFN FLOPs (currently only Ex 5).
- The model as a *causal system* mapping ids₁..ₜ → P(x_{t+1}): not framed; a 6-line test (change `ids(8)`, show logits rows 1–7 unchanged, heatmap |Δlogits|) proves causality end-to-end.
- Why logits → softmax → loss: L199 says what, not why (log-odds, max-entropy/Gibbs normalisation, CE = code length). Two sentences + links to L02/L03.
- Full architecture diagram: ASCII only.

**4. EE / controls / IT reframing.**
- *Exact:* GPT is a causal, nonlinear, finite-memory (T_max) system; the mask is the h[n]=0, n<0 constraint applied per layer. The Δlogits causality test is the figure.
- *Exact with weight tying:* logits = ⟨h_f, e_v⟩ for every v — a correlation-receiver / matched-filter bank followed by Gibbs normalisation; temperature is 1/kT (loose). Good hook for Lesson 21.
- *Controls:* `semilogy` ‖H^{(l)}‖ for l = 0..N as an open-loop gain plot; tie to the 1/√(2N) init in the sidebar.
- IT: keep L300; add one sentence on Chinchilla's ≈20 tokens/param as "bits of data per bit of budget".

**5. Rustlab 0.3.7.**
- ```mermaid architecture diagram with shapes on edges (replace L21–25); optional second one for block internals.
- `run gpt_lib.rlab` for `block_fwd` and PE.
- `<!-- grid: 2 -->` bar charts: toy vs GPT-2 small breakdown.
- signed `imagesc(logits)` or `heatmap` with position labels; drop `- min`.
- New example block: causality test.

**6. Defects.** None numerical (3200 / 49,728 / 198,912 / 205,440 / 124,402,944 confirmed; GPT-2 medium 354,724,864 / 98,304 confirmed by running `param_count.rlab`). Narrative: L272 promise unfulfilled; L296/L360 overclaim.

**7. Severity: light polish → moderate rewrite** if FLOPs, causality test and the diagram are added (recommended; they are the checklist items).

---

## Lesson 15 — Backpropagation

**1. Covers now.** Notation table with row convention (L17–28); chain rule; 2-layer MLP backward by hand + FD check; linear-layer triple; softmax+CE p − 1_y + FD; attention head 4-step backward + FD at W_Q; gradient flow through 12 random layers (heatmap); connections. Figures: 1 heatmap (3×12). No computational-graph figure.

**2. Clarity problems (ranked).**
1. No computational graph is ever drawn. Attention's fan-out/fan-in (X feeds Q, K, V; gradients sum at X, L224) is exactly where a DAG picture is needed.
2. rustlab-quirk comments inside teaching code (L64–65, L73–75, L134, L181: "vector-vs-1×N mismatch", "M(1) trap"). Authoring notes, not lesson content — one callout or delete.
3. L326–331 attributes growth/decay to "spectral radius". The demo multiplies by *independent* random matrices, where E‖gJ‖² = ρ²‖g‖² (RMS gain / singular values); spectral radius only governs the tied-weights asymptotic case and can be <1 while the norm grows transiently (non-normal J). An EE reader will object. Say "gain".
4. L380 "each gradient component is a Shannon-like quantity … high mutual information with the loss" — not true; replace (see §4).
5. Heatmap for three curves: a `semilogy` of ‖g_k‖ vs k is clearer than a 3×12 heatmap normalised to a global min; L368 currently has to narrate the numbers.
6. L253–293 is 41 lines; acceptable, but theory Steps 1–4 sit above and code below — interleave.
7. Lesson 14's "What's next" promised backprop "through softmax, attention, FFN, and LN"; LN and GELU backward are never derived. Derive LN backward (the one non-obvious case) or fix the promise and forward-ref Lesson 24.

**3. Missing / thin.**
- Adjoint/costate equivalence: absent. Reverse vs forward mode: never mentioned (why reverse mode for a scalar loss: one backward pass vs n_params forward passes).
- Residual Jacobian I + f' is stated (L331, L378) but not in the demo; Ex 4 asks the student. Put the "ρ=0.7 + identity" curve inline.
- Minibatch sum-vs-mean convention only in Ex 5.
- FD check present and good; add the relative-error form and why ε ≈ 10⁻⁵ (√machine-eps).

**4. EE / controls / IT reframing.**
- *Exact, worth stating as a theorem:* backprop is the discrete adjoint / costate recursion. For x_{k+1} = f_k(x_k, θ_k), L = ℓ(x_N): λ_N = ∂ℓ/∂x_N, λ_k = λ_{k+1} ∂f_k/∂x_k, ∂L/∂θ_k = λ_{k+1} ∂f_k/∂θ_k. That is L42–46 with ā_k renamed λ_k (Bryson–Ho 1969; LeCun 1988). Figure: mermaid two-lane diagram, forward state lane left→right, adjoint lane right→left, parameter gradients as rungs. Computation: relabel the MLP code with λ and show it is unchanged.
- *Exact:* backward through y = xW is the adjoint operator x̄ = ȳWᵀ; for an LTI filter the adjoint is time-reversal (matched filter). Gives L124's slogan a name.
- *Exact:* depth = cascade of per-stage gains; the residual's identity branch is a unity feed-forward path guaranteeing gain ≥ 1 − ‖f'‖. Figure: `semilogy` four curves (ρ = 0.7, 1, 1.3, 0.7+I).
- IT: p − y is the score function; E[score] = 0 at the true model is why the logit gradient sums to zero (`softmax_ce_grad.rlab` L49–53 already shows it); Adam's v_t (next lesson) is the diagonal empirical Fisher. Replace L380 with this.

**5. Rustlab 0.3.7.**
- ```mermaid: MLP graph (L36), attention-head graph (Steps 1–4), forward/adjoint two-lane diagram.
- `semilogy` replaces the heatmap (or `<!-- grid: 2 -->` both); the `g_min` normalisation and comment at L360–362 are obsolete with signed heatmaps.
- `<!-- exercise -->` for Ex 4 with the identity-branch run inline (solution directive pending upstream md fix).
- `run` the shared attention forward from Lesson 13's lib.

**6. Defects.** All numbers match (L = 1.1655; 1.10e-11; 3.60e-11; 1.55e-12; ratios 0.037 / 0.89 / 25.8; "27×" = 1/0.037). "Spectral radius" imprecision (above). L14 over-promise (above). Table row "≈ 1e-7 or smaller" is loose but true.

**7. Severity: moderate rewrite** — derivations correct and well checked, but the adjoint framing (the single biggest win for this audience), the graph picture, and the reverse-mode motivation are missing, and quirk comments clutter the code.

---

## Lesson 16 — The AdamW Optimizer

**1. Covers now.** SGD on a κ=200 bowl; momentum vs SGD (κ=20); Adam with bias correction; coupled vs decoupled decay with a fixed-point demo; three-trajectory overlay; defaults incl. the β₂=0.95 LLM nuance; IT framing. Figures: 2 (log-loss SGD vs momentum; trajectory overlay). No contours.

**2. Clarity problems (ranked).**
1. The momentum loss curve (plot 1) visibly oscillates with period ≈7 steps and nobody says why. It is the underdamped heavy-ball pole: z² − (1+μ−ηa)z + μ = 0 with μ=0.9, η=0.01, a=20 → |z| = √μ = 0.949, angle 26.4°, 13.7-step period in θ₁, halved in the loss (θ²). Verified numerically. The checklist's damping-ratio item is already in the figure.
2. Unexplained bowl switch: a=200 at L38, a=20 at L81, back to a=200 at L150. Heavy ball is stable on a=200 at η=0.009 (needs ηa < 2(1+μ) = 3.8), so either keep one bowl or say why.
3. Trajectory overlay (L274–284) has no loss contours; the "ravine" is invisible.
4. L286 quotes ‖θ‖ ≈ 0.024 for AdamW; nothing prints it (I confirmed 0.02399). Print it.
5. L307 "decoupled decay corresponds to a Gaussian prior N(0, 1/λ)" contradicts L193's own point that AdamW is *not* the L2/MAP solution. Reword as "a leak toward zero applied outside the preconditioner".
6. L34 asserts η < 2/a without derivation, and Lesson 06 has no bound to point to (its Ex 1 is empirical). Two lines: θ ← (1 − ηa)θ, need |1 − ηa| < 1.
7. `v` is the velocity in §Momentum (L75) and the second moment two sections later (L140).

**3. Missing / thin.** Momentum as a one-pole IIR low-pass (time constant, DC gain) — only "steady-state gain 1/(1−μ)". Heavy ball as a second-order system with damping — absent. Second-moment normalisation as AGC / normalised LMS — absent (L306 "SNR-like" is closest). Adam near convergence as sign descent with an η-sized limit cycle (why the LR must decay) — absent, though Lesson 18's grad-norm plot shows it. ε and defaults: fine.

**4. EE / controls / IT reframing.**
- *Exact:* m_t = βm_{t−1} + (1−β)g_t is H(z) = (1−β)/(1 − βz⁻¹): pole at β, DC gain 1, time constant −1/ln β ≈ 1/(1−β) steps, 3 dB at ω ≈ 1−β rad/step. Heavy-ball v = μv + g is the same filter with DC gain 1/(1−μ). Bias correction divides out the step-response deficit 1 − βᵗ. Figure: `freqz` magnitude for β ∈ {0.9, 0.99, 0.999} beside the step response (`grid: 2`).
- *Exact:* heavy ball on a mode with curvature a is second-order; complex poles have |z| = √μ, critical damping at ηa = (1 − √μ)². Figure: pole locations for a ∈ {1, 20, 200} via `roots`, or `saveanim` of the trajectory.
- *Exact:* 1/√v̂ is diagonal preconditioning = per-coordinate AGC / NLMS — every coordinate's step has unit RMS regardless of gradient scale. The "≈ inverse Hessian" reading is loose (it is √ of the diagonal empirical Fisher).
- *Exact:* weight decay is a leak term (leaky integrator); "decoupled" = leak outside the preconditioner.
- Derive η < 2/λ_max(H) as eigenvalues of I − ηH inside the unit circle here; reuse in Lesson 17.

**5. Rustlab 0.3.7.** `contour` under both trajectory plots; `freqz`/`bode` of the EMA; `roots` for heavy-ball poles; `stem` for the EMA impulse response; `saveanim` of the three optimisers (the canonical use case); `<!-- grid: 2 -->` for loss-curve + path pairs.

**6. Defects.** Expected-Outputs row "Bias correction at t = 100 ≈ 1 (negligible)" is false for β₂: 1 − 0.999¹⁰⁰ = 0.095, a 10× rescale, contradicting L146. Notebook vs script mismatch: the notebook overlay is noiseless, 60 steps; `optimizer_comparison.rlab` adds σ=0.5 noise for 80 steps (Adam 0.0706, AdamW 0.0352) while L323 implies they are the same demo. Everything else matches (2.7035, 2.325, 0.0533, 45×, 0.0169, (0.909, 2.500), (0.967, 4.740)).

**7. Severity: moderate rewrite** — honest and correct, but this is the lesson where the EE reframing is *exact* and cheap, and the text stops at "EMA".

---

## Lesson 17 — Learning Rate Scheduling

**1. Covers now.** Warmup rationale via Adam's v̂; cosine formula and endpoints; why not step/exponential; schedule plot; SGD demo of high / low / scheduled on a noisy quadratic with steps-to-floor; cheat sheet of published values; IT framing. Figures: 2 (schedule; three log-loss curves).

**2. Clarity problems (ranked).**
1. The experiment does not test the thesis. Warmup is motivated by Adam's v̂ (L10, L21–26), but the demo (L92–167) uses plain SGD, and the schedule's peak 0.18 is under the stability limit 0.2 — a constant η=0.18 would match or beat it. The figure demonstrates the SGD stability bound (Lesson 16 L34), not warmup.
2. Cross-lesson inconsistency: L31 and L203 anchor T_w to 1/(1−β₂) ≈ 1000, while Lesson 16 L295 says LLMs use β₂ = 0.95 (horizon ≈ 20 steps). Honest version: warmup length is empirical; v̂ is one of several reasons (early sharpness, never-updated embedding rows, Post-LN instability — Lesson 12 L191 already cites Xiong 2020).
3. 76-line block (L92–167). A `run_sgd(eta_of_t)` returning the curve (≈12 lines), three one-line calls, steps-to-floor in `<!-- details -->`.
4. "Bayes risk" (L213) undefined.
5. Plot 2 is dominated by the clamped red line at 10⁴; the informative content is a thin band near 0.

**3. Missing / thin.** Gain-scheduling framing: absent. Stability bound: used (2/10) but not derived or linked. Adam's step-1 mechanics — m̂₁/√v̂₁ = sign(g) exactly, so at t=1 every coordinate moves by η_max regardless of gradient size (`adam_step.rlab` L41–44 already prints this) — is the clean justification for warmup and is absent. Modern alternatives (constant + cooldown / WSD, inverse-sqrt): one sentence.

**4. EE / controls / IT reframing.**
- *Exact:* constant-η GD on a quadratic is LTI: θ_{t+1} = (I − ηH)θ_t, stable iff |1 − ηλ_i| < 1. A schedule makes it LTV: per-mode gain Π_t |1 − η_tλ_i|. Figure: plot |1 − η_t·10| and |1 − η_t·1| vs t for the three runs with `hline(1)` — the red run sits above 1 forever, warmup starts below 1. This *is* gain scheduling and should replace the clamped log-loss plot.
- *Exact (stochastic):* with gradient noise σ² the steady-state per-mode variance ∝ ησ²/(2 − ηλ); decaying η lowers the noise floor — L86 says this in words; give the formula and a `loglog` of final loss vs η_min.
- *Loose but right vocabulary:* warmup = soft-start (limit inrush while the plant state is unknown).
- IT section L206–216: already an SNR framing; keep, drop "Bayes risk".

**5. Rustlab 0.3.7.** `<!-- grid: 2 -->` η_t beside the contraction factor; `semilogy` for losses (drops the manual clamp); `saveanim` of the scheduled run on `contour` with η_t in the title; `<!-- exercise -->` for Ex 1 and 3; `run` a shared schedule function so Lesson 18 (L186–191) stops re-implementing it.

**6. Defects.** Numbers match (127 / 36 / 3.5× / 1.0006 / 1.0018); cheat-sheet values check against nanoGPT, GPT-3, LLaMA-1/2. Narrative defects are items 1–2 above.

**7. Severity: moderate rewrite** — short and formula-correct, but the central experiment tests the wrong thing and the natural (LTV/gain-scheduling) framing is absent.

---

## Lesson 18 — The Training Loop

**1. Covers now.** 24-parameter embedding+head model; forward; backward + FD check; five-step loop; *description* of `train_loop.rlab`'s three diagnostics; inline 180-step mirror (train/val, grad-norm plots); empirical-floor sanity check (24 ln2/49); connections; clipping sidebar with commented-out code. Figures: 2 inline. `train_loop.rlab`'s three SVGs and `overfit_demo.svg` are not embedded.

**2. Clarity problems (ranked).**
1. "Reading the Diagnostics" (L136–149) describes three plots the reader cannot see. The overfitting signature promised in the objectives (L11) is never shown in the notebook; `overfit_demo.svg` exists only under `lessons/`.
2. The grad-norm prose contradicts its own figure. L128/L147 say "decays smoothly … brief upticks"; both the 180- and 600-step plots are a sawtooth with 1–2-decade spikes from step ≈20 to ≈300. Ex 4 says the grad-norm peak aligns with the LR peak — it does not (peak ≈13 vs T_w=30 inline; ≈25 vs T_w=60 in the script); the peak sits at the steepest loss descent, before η_max.
3. 58-line loop block (L155–212) with the Adam update written out twice. An `adamw_step(θ, m, v, g, t, η)` helper makes "one step" visible and halves the block; corpus/buffers go to hidden setup.
4. In plot 1 the train curve visibly "ends" at step ≈40 because the green floor `hline` is drawn over it.
5. "AdamW" in title/objectives but λ = 0 (L262): the reader implements Adam. Run λ = 0.1 or say so up front.
6. Clipping code (L289–297) is comment-only and uses untested `sum(M, "all")`. Make it live and print how often it fires.
7. Loop is full-batch (49 pairs per step) while the recipe says "sample a minibatch" (L111).

**3. Missing / thin.** Minibatch noise (described, not shown); checkpoints, logging cadence, early stopping (absent); clipping as a limiter (sidebar, not framed or run); under/over-fitting runs (table only); closed-loop framing (absent); the *reason* for the grad-norm sawtooth (absent, and it is the best link back to 16/17).

**4. EE / controls / IT reframing.**
- *Exact:* training is a discrete-time feedback loop — plant θ, noisy sensor (minibatch loss/gradient), controller = optimiser (momentum is the integral term, 1/√v̂ is AGC, decay is a leak), actuator limiter = clipping, gain schedule = LR, error signal = p − y (Lesson 15 L170 already calls it the prediction error). Figure: one mermaid block diagram with those labels — the single most useful figure for this audience.
- *Exact:* g·min(1, c/‖g‖) is a vector-norm limiter (magnitude saturation, direction preserved); plot output norm vs input norm with the knee at c.
- *Exact:* the sawtooth is Adam's limit cycle: near the optimum v̂ tracks the shrinking gradient, so each coordinate's step stays ≈ η_t in size and overshoots/reverses (relay-controller behaviour). Amplitude ∝ η_t, so log‖g‖ tracks the cosine and floors at ≈10⁻⁷ once η_min/ε-limited. Figure: log‖g‖ and log η_t on a shared step axis — the correlation L149 wants to show.
- IT: already strong. Add the val floor 5 ln2/9 = 0.385 explicitly (it equals the observed 0.3851 exactly; the text explains the weighting but never states the number). Overfit demo: val = −log P(3|1) is the cost of a transition with zero training support — the unseen-event term from Lesson 03.

**5. Rustlab 0.3.7.** ```mermaid closed-loop diagram; `run train_lib.rlab` (forward/backward/adamw_step/schedule, shared with 17 and 24); `<!-- grid: 3 -->` train/val, grad norm, η_t on one step axis; `semilogy` for grad norm (drops the "from step 1" workaround); inline `overfit_demo` or `![[overfit_demo.svg]]`; `<!-- exercise -->` for Ex 2 with a constant-LR run; live clipping with `tic/toc`.

**6. Defects.** Ex 4 premise false; "decays smoothly" false. Expected Outputs match the script (1.1201, 0.3395, 0.3851, 9.39e-8, 0.2773, 6.208 all confirmed). "Initial gradient norm ≈ 0.18" is never printed (plausible from the plot) — add a print. L102 "≈10⁻⁹" is conservative (actual 5.4e-12).

**7. Severity: moderate rewrite** — the numbers are impeccable, but the lesson shows one healthy run and *tells* about everything else, the grad-norm story contradicts its own plot, and the feedback-loop framing is absent.

---

## Cross-lesson summary (13–18)

**Notation drift.** Gradients are ā in L15 but `dE/dW/g` in 16–18; η vs `eta` vs "LR"; step index t (16–18 prose) vs k (16 code) vs ℓ (layers); `d_model` / `d` / `d_emb`; `v` is momentum velocity in L16 §Momentum and the second moment two sections later. Row-vector convention is declared in L15 but L13 L168 writes column form. A "step" is one gradient in L16 and a full-batch epoch in L18.

**Repeated pain points.** (1) The block forward is copy-pasted five times (L13 inline ×2, L13/L14 scripts, L14 hidden) — `run` resolves it. (2) Stale 0.3.6 workaround comments in 13, 14, 15, 18 (signed heatmap, subplot TODO, log-ratio normalisation, "from step 1"). (3) rustlab-quirk comments inside teaching code (15, 18). (4) Prose describing figures that are absent (18) or unreadable (13). (5) Blocks over 40 lines in 13, 14 (hidden), 15, 17, 18. (6) No diagrams anywhere — ASCII in 13/14, none in 15/18. (7) `optimizer_comparison.rlab` and the L16 notebook overlay differ (noise) while the table implies parity.

**Does 15→18 read as one feedback-control narrative?** No — four competent recipes. L15 ends on "credit assignment"; L16 opens on "loss surfaces are not paraboloids"; L17 motivates warmup by Adam's v̂ and tests it with SGD; L18 says "five lines" and shows one clean run. The connective tissue an EE audience expects — error signal → adjoint → controller (filter + AGC + leak) → gain schedule → closed loop with a limiter and a noisy sensor — is never drawn, and the one place the material itself exhibits the coupling (Adam's limit-cycle sawtooth in L18's grad norm, which is *why* L17's decay exists) goes unexplained.

**Three highest-leverage changes.**
1. One mermaid "training as a feedback loop" diagram, introduced at the end of L15 and re-shown in L18 with each block filled in as it is taught (adjoint = 15, filter/AGC/leak = 16, gain schedule = 17, limiter + sensor noise = 18). Cheap, and it turns four lessons into one story.
2. L16 → L17: replace "EMA" with the one-pole filter (`freqz`), heavy-ball poles (`roots`; the oscillation already in the figure), AGC framing; derive η < 2/λ_max in L16 and reuse it in L17 as the LTV contraction-factor plot that replaces the current SGD demo. Fix the two factual slips (β₂ correction at t=100; MAP-prior wording) and the L17/L16 β₂ inconsistency.
3. L13 → L14: mermaid block and architecture diagrams with shapes on arrows, the two-mixing-axes perturbation demo, the causality test, the FLOPs estimate, signed-heatmap grids, and a shared `run` library to remove the duplication.
