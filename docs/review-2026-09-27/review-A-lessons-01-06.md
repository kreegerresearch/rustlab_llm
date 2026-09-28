# Review A — Lessons 01–06 (read-only)

Sources read in full: `notebooks/01..06-*.md`; skimmed `book/01..06-*.md` (captured outputs + all 17 referenced SVGs rendered to PNG and inspected) and `lessons/01..06/*.rlab`. Numbers quoted below (Hessian eigenvalues, chain eigenvalues, entropies) were re-computed with rustlab 0.3.7 in the scratchpad.

---

## Lesson 01 — Tokens & Text Encoding

**1. Covers now.** Token definition, sorted vocabulary + index table for `"to be or not to be"`, relative frequencies, one-hot vectors and orthogonality, the T×|V| matrix X for `"hello"`, an IT section (source-coding bound, unigram entropy 2.594 bits vs log2 7 = 2.807). Figures: 2 (frequency bar chart, one-hot heatmap).

**2. Clarity problems (ranked).**
- Two corpora/vocabularies are used mid-lesson (`"to be…"` for frequencies, `{e,h,l,o}` for one-hot, lines 39/102). One corpus throughout.
- The thesis "text → integers" (line 3) is never executed: `encode`/`decode` exist only as prose (line 39) and exercise 2. No block ever turns a string into `seq`; Lesson 05 line 49 then hard-codes its sequence.
- Line 19 is editor meta-text ("This section is pure reference — every later section pairs `### Theory` with `### Example`") leaking into the student page (also 02:25, 03:19, 04:19).
- Figure 1: the first bar's label is the space character, so it is invisible; legend reads "bar". Label it `␣`/`(sp)`.
- Figure 2 and every categorical heatmap in 01/04/05: row labels sit on the top *edge* of each cell, not the centre, so "the bright cell on each row sits under the column" (line 150) requires guessing which label owns which row (renderer quirk — worth an upstream note).
- The IT section's headline number is computed inside a `${…}` template (line 161) — no visible code. Make it a block and define an `entropy_bits(p)` helper that 02 and 05 reuse.

**3. Missing / thin.** Executed encode/decode round trip; sparsity (exercise-only — print the 1/|V| density once); the lookup teaser `X * eye(4) == X` that makes 04's "one-hot × matrix" land early; the *source* framing: the corpus as a realisation of a discrete source X_t over alphabet 𝒱, and that 2.594 bits is the **memoryless** (zeroth-order) floor which 05 beats by conditioning; symbol rate — no mention that tokenisation trades symbols/s against bits/symbol while bits/char is invariant (the fact Lesson 19 needs).

**4. Reframing.**
- Exact: tokeniser = source alphabet; f_i = first-order marginal; H(f) = memoryless source-coding bound. Frame the course as a staircase of floors: log2|V| = 2.807 → H(unigram) = 2.594 → H(X_{t+1}|X_t) (05) → H(X_t | past) (attention). One bar chart of these floors, extended in 05 and 22.
- Exact: one-hot = standard basis / unary code (redundancy 1 − log2|V|/|V|) — already there; keep.
- Exact (save for 04): one-hot × E is a codebook lookup (vector quantisation).
- Symbol rate: one computed example (18 chars × 2.594 bits/char vs 6 words × word-level entropy) shows bits/message is what is conserved.

**5. 0.3.7.** Mermaid pipeline at line 19 (text → chars → ids → one-hot → [04] embedding); `<!-- grid: 2 -->` for the two figures; `stem(freqs)` is the honest form for a PMF; `<!-- exercise -->` for the five exercises (solutions pending upstream fix); a shared `_setup.md` with `entropy_bits`.

**6. Defects.** Expected-Outputs table matches book. `char_frequencies.rlab` line 42 uses numeric-x `bar(freqs, …)` while the notebook uses labelled bars (minor divergence). Line 165 says perplexity is "2^H in disguise" while 05 computes e^L — base drift (see summary).

**7. Severity: light polish** — add an executed encode/decode block and the "floors" framing; the rest is cosmetic.

---

## Lesson 02 — Probability & Softmax

**1. Covers now.** Softmax construction, max-shift stability callout, temperature table + 3-temperature example, entropy in bits at four temperatures, uniform / near-deterministic checks, IT section (source coding, Boltzmann, max-ent asserted). Figures: 2 (3×1 stacked line plots; entropy bars + `hline`).

**2. Clarity problems (ranked).**
- Figure 1 draws a 4-point PMF as continuous lines on a **0–3** x-axis while the prose says "token 1" (line 75) and rustlab is 1-based. Use `stem`/`bar`, one overlaid panel or `<!-- grid: 3 -->` instead of 1100 px of stacked panels.
- Objective "derive softmax from first principles" (line 8) is not delivered: lines 29–37 are a construction; the max-entropy derivation that would deliver it is a single assertion (line 179). A six-line Lagrangian (max H s.t. Σp = 1, Σp_i z_i = μ ⇒ p ∝ e^{βz}) is exactly what this audience wants.
- The stability callout (line 39) never names **log-sum-exp** or the **partition function** Z — the two names 03 (L = −z_c + LSE(z)) and the Boltzmann paragraph depend on.
- Figure 2's y-axis auto-starts at 0.8, exaggerating the T = 0.5 → 1.0 jump; legend "bar".
- `eps` inside `log2` (lines 116–119) needs the one-line convention 0·log 0 = 0.
- T = 5.0 is computed (line 127) but absent from Figure 1 — the flattening toward uniform is the punchline and is not shown.

**3. Missing / thin.** Logits as log-odds (z_i − z_j = log p_i/p_j — absent; shift invariance is exercise 1 only); T → 0 / T → ∞ limits are a table (line 51) with no derivation or numeric demonstration; partition function; max-ent derivation; two-class softmax = logistic sigmoid (one line); the softmax Jacobian ∂p_i/∂z_j = p_i(δ_ij − p_j) (one line; makes 03's gradient trivial).

**4. Reframing.**
- Exact (comms): if z_i = log p(evidence | i) + log prior_i, softmax(z) is the Bayesian posterior; for two classes z_1 − z_2 is the log-likelihood ratio and softmax is a soft-decision demapper. Temperature scales the LLR exactly as noise variance scales it in a demapper. One computation: `z(1)-z(2)` and `log(p(1)/p(2))` both print 1.000 (verified).
- Exact (stat mech): name Z(T) = Σ e^{z_i/T}; LSE = log Z.
- Exact: softmax is a soft-argmax; T → 0 recovers the hard comparator. Figure: entropy vs T as a continuous curve on a log-T axis (`semilogx`, T ∈ [0.05, 50]) with `yline` at 0 and log2 4 — replaces the four bars and shows both limits.
- Avoid "temperature = gain" (loose, misleads).

**5. 0.3.7.** `stem` for PMFs; `<!-- grid: 4 -->` for T = 0.5/1/2/5; `semilogx` entropy curve; `saveanim`: softmax bars morphing from argmax to uniform as T sweeps (~40 frames) — the most convincing temperature visual; slider on T under `watch`.

**6. Defects.** All 10 Expected-Outputs entries match book. Notation collision: T = temperature here, T = sequence length in 01/03/04/05. `entropy.rlab` leaves `hold("on")` open (script only).

**7. Severity: moderate rewrite** — maths is correct, but the lesson's own objective (derive softmax) and the checklist's log-odds / max-ent / limits are missing, and both figures need re-forming.

---

## Lesson 03 — Cross-Entropy Loss

**1. Covers now.** CE definition → −log p̂_c, reference values, loss curve, MLE (one paragraph), gradient dL/dp̂_c, 2-D loss surface in logit space, IT section (CE = H + KL, code length, bits/nats, MDL), label-smoothing sidebar. Figures: 2 (loss curve, 3-D surf).

**2. Clarity problems (ranked).**
- **"Gradient of the Loss" (lines 87–100) reaches the wrong conclusion.** It differentiates w.r.t. p̂_c and claims a "100× stronger signal … forces the model to avoid confidently wrong predictions". The optimiser sees ∂L/∂z = p − y, which is **bounded by 1**; the lesson's own "linear wall" paragraph (line 143) says so without stating the formula. For an audience that will ask "bounded or not?", this teaches the wrong intuition. Replace with the one-line derivation from L = −z_c + LSE(z) and print p − y for the 3-class example.
- MLE section (lines 77–85) is one equation; the product → log → sum → ÷T steps are skipped.
- The surf is unreadable in the static render (no axis labels survive; only "x=-3.000"), so "near-zero when z_1 dominates" cannot be checked. `contourf(Z1, Z2, L_surface, 20)` shows the linear wall as evenly spaced parallel contours — the exact point the text makes.
- Everything is in nats while the IT section reasons in bits; add a `/log(2)` column to the reference table.
- Line 181 uses an undefined symbol H_unif(p̂); it is H(u, p̂), the cross-entropy against uniform u. Line 19 boilerplate.

**3. Missing / thin.** ∂L/∂z = p − y (not deferrable here — the section exists and draws a conclusion); NLL/MLE derivation; KL never computed numerically (two lines: KL between 02's T = 1 and T = 2 softmaxes, before it returns in 24/DPO); convexity of −z_c + LSE(z) is never stated, yet 06 says "LM losses are non-convex" — the reader should learn convexity is lost in the network, not the loss; "why CE and not MSE" is never asked (answer below).

**4. Reframing.**
- Exact: CE = expected code length under a mismatched code; KL = excess bits — already strong.
- Exact: **MSE is cross-entropy under a Gaussian likelihood** (−log 𝒩(y; ŷ, σ²) = (y − ŷ)²/2σ² + c). One equation at the end of 03 turns 06's MSE from "for intuition" into the same objective under a different noise model.
- Exact: p − y is a bounded **error signal** — softmax + CE is a soft-decision detector whose error saturates at ±1 (contrast the unbounded −1/p̂_c). Figure: p̂_c and ∂L/∂z_c = p̂_c − 1 vs z_c (sigmoid-shaped, saturating).
- Loose (one sentence): LSE as the smooth max, the wall as a soft hinge.

**5. 0.3.7.** `contourf` for the surface (or `<!-- grid: 2 -->` surf + contour); `semilogx(p_hat, loss)` makes the log(1/p) law a straight line; mermaid: y, z → softmax → CE → scalar, with the p − y backward arrow — first appearance of the forward/backward pair.

**6. Defects.** Table matches book (L_surface max ≈ 8.0 vs 8.007). Orphan tracked plot `book/plots/03-cross-entropy-loss/plot-1-8d6022f8.svg` is unreferenced. Line 165's BPB/BPC figures are unsourced (footnote). The gradient section is the one substantive error.

**7. Severity: moderate rewrite** — replace the gradient section, add the MLE steps and the MSE/Gaussian bridge, swap surf for contourf.

---

## Lesson 04 — Embeddings & Similarity

**1. Covers now.** One-hot limitations, E ∈ ℝ^{|V|×d} and lookup, seeded random 8×6 E + heatmap, cosine similarity with a 4×4 S (16 calls, then matrix form), similarity heatmap, analogy arithmetic, parallelogram plot. Figures: 3 (imagesc of shifted E, cosine heatmap, parallelogram).

**2. Clarity problems (ranked).**
- Convention flips inside one section: line 33 $\mathbf E^\top\mathbf e_i = \mathbf E_i$ (column) then line 39 $\mathbf H = \mathbf X\mathbf E$ (row); code uses `e3 * E` (row). Pick row: $\mathbf e_i\mathbf E = \mathbf E_{i,:}$.
- Lines 79–83: stale 0.3.6 workaround (`E - min(min(E))`, comment citing the issues doc) yields a heatmap whose colourbar (0–0.443) means nothing and whose title says "E − min(E)". Signed heatmaps now render — plot E with token/dim labels.
- Lines 133–150: sixteen `cos_sim` calls immediately followed by the 4-line matrix form (164–170). Keep three scalar calls plus the matrix form.
- Figure 2's colourbar autoscales to 0.509–1.0, so king/woman (0.51) renders black as if orthogonal, contradicting the [−1, 1] table (line 94). No `clim` exists; state the range in a `<!-- caption -->` or print S beside it.
- Figure 3: points are unlabelled (which dot is king?), legend lists "femininity"/"royalty" twice, and line 236 spends ~200 words describing what the figure should show. `quiver` arrows for the two displacements; name the points in caption/table (no text primitive exists).
- Line 76 "all rows look similar" vs a noise heatmap — say "no structure". Line 205 "not programmed" appears directly after hand-programmed vectors — say it is by construction here.
- imagesc axes are 0-based (0–8, 0–6) vs "row 3" in prose; `embedding_matrix.rlab` line 44 says rows are "0..vocab_size−1".

**3. Missing / thin.** Dot product as correlation / matched filter (absent); dimension choice (one bullet, no argument); norm effects (one sentence, no demo — scale `king` ×10, cos unchanged, dot not); learned vs fixed (nothing is learned and the lesson never says when E is trained — point to 18); objective "identify patterns in a learned heatmap" cannot be met on random data — drop it or `load` a trained E saved by 22.

**4. Reframing.**
- Exact: a·b is a correlator; cosine is normalised cross-correlation; S = ÊÊᵀ is a correlation (Gram) matrix; a matched filter's output at the sampling instant is an inner product with a template — so "query · key" in 08 is a bank of matched filters. Plant it here with one figure: `king` correlated against the four templates as a 4-bar filter-bank output.
- Exact: E is a codebook (VQ); one-hot × E is codebook lookup; d is the codeword dimension.
- Exact and cheap: random vectors in ℝ^d are nearly orthogonal — `histogram` of |cos| for 1000 random pairs at d = 2, 8, 64 answers "how to choose d" and connects to 01's orthogonal one-hots.
- Keep "semantic axes" explicitly labelled as a toy.

**5. 0.3.7.** Signed `heatmap(dims, tokens, E)` replacing 79–83; `<!-- grid: 2 -->` for heatmap + parallelogram; `quiver` for the analogy arrows; `histogram` for the random-cosine experiment; mermaid: id → one-hot → E → h with shapes on arrows.

**6. Defects.** Table matches book. The workaround comment is now factually wrong for 0.3.7. 0-based comment in the .rlab.

**7. Severity: moderate rewrite** — figures 1 and 3 need redoing, the convention flip must be fixed, and the correlation/matched-filter and d-choice content the audience expects is absent.

---

## Lesson 05 — The Bigram Language Model

**1. Covers now.** LM problem, Markov assumption, count matrix C, row normalisation, add-one smoothing, per-row entropy, C/P heatmaps, CDF sampling + 12-token generation, CE/perplexity on the corpus, IT section (conditional entropy, chain rule, compression). Figures: 3 (two heatmaps, 3×1 bar panels).

**2. Clarity problems (ranked).**
- Generated text is never shown: the book prints `[1×12] 1 2 1 2 3 2 3 2 ... (12 total)` — truncated by the renderer and never decoded to `ababcbcb…` (PLAN.md line 74 promises students see text). Decode via `tokens` and print a string.
- Row normalisation hard-codes three columns (lines 78–80, 107–109). `M(i,:) = vec` works since 0.3.4 and `P = C ./ sum(C, 2)` is one line (04 already uses `./ sum(…, 2)`). The hard-coding also silently breaks exercises 1/3 if |V| changes.
- Row entropies (line 137) and corpus CE (line 222) are never connected by H(X_{t+1}|X_t) = Σ_i π_i H(P_{i,:}); line 251 writes "≈" without saying what π is.
- Line 196 notes "b at every even position" without the reason (spectral, below).
- "Row-stochastic" is never used; smoothing is not shown to *raise* training CE (0.679 vs 0.347 nats — two lines, a bias/variance point).
- 3×1 bar panels → `bar(P)` grouped or `<!-- grid: 3 -->`.

**3. Missing / thin.** Stationary distribution / eigenvector (absent); entropy rate (absent); add-k for general k; MLE claim (line 202) asserted, not derived (3-line Lagrangian, same shape as 02's max-ent); the "floors" staircase (log2 3 = 1.585, H(unigram), 0.5 bits) printed side by side.

**4. Reframing (this is where the controls story is exact).**
- Exact: π_{t+1} = π_t P is a discrete-time LTI system on the simplex; row-stochastic ⇒ spectral radius 1, eigenvalue 1 with left eigenvector π. For this corpus `eig(P')` = {1, 0, −1} (verified): the −1 pole on the unit circle is a marginally stable period-2 mode — that is why b lands on every even position and why the empirical unigram [3,4,2]/9 ≠ π = [¼, ½, ¼]. Smoothing pulls the −1 pole inside the circle; the chain becomes ergodic and mixes at rate |λ₂|^t. One computation: `[V,D] = eig(P')`, check `pi*P`, then `abs(eig(P_smooth'))`. One figure: `stem` of π_t = e_a P^t for t = 0..8 for P (oscillates) vs P_smooth (settles). Note `P^k` is element-wise in 0.3.7 — iterate with a loop.
- Exact: entropy rate = Σ π_i H(row i) = 0.5 bits — ties Row Entropy to the CE section.
- Exact: inverse-transform sampling — name it.
- Exact: this is Shannon 1948's own first-order Markov source construction — cite it; the audience knows the paper.

**5. 0.3.7.** Mermaid `graph LR` state diagram (a→b 1.0, b→a 0.5, b→c 0.5, c→b 1.0) at the count section — it *is* a signal-flow graph; `eig` for π; `stem` for π_t; `<!-- grid: 2 -->` for C/P; `heatmap` of P_smooth (currently only printed); `saveanim` of π_t evolving; `<!-- details -->` for long prints.

**6. Defects.** Table matches book. Truncated `draws`/`generated` prints in book are reader-facing. `bigram_sampling.rlab` hard-codes `log_probs` and line 7 references `bigram_counts.r` (wrong extension). Line 255 mixes 2^{0.5} with a ppl computed as e^{L nats} — equal here, but say so.

**7. Severity: moderate rewrite** — what is present is correct; the lesson lacks its best content (spectral view, entropy rate) and its promised output (text).

---

## Lesson 06 — Linear Layers & Gradient Descent

**1. Covers now.** Affine layer definition, MSE on x = [1..4], y = 2x, analytic expansion, 40×40 heatmap + surf, gradient derivation, 200 GD steps at η = 0.05, loss curve, (w,b) path, LR table, one hand-checked step. Figures: 4 (imagesc, surf, loss vs step, path scatter).

**2. Clarity problems (ranked).**
- **Broken hand-off.** 05's What's-next (line 308) and PLAN.md line 75 promise "replace the count table with a learned linear layer … same optimum the count estimator finds". 06 never touches tokens; it fits y = 2x.
- **Presented as converged while visibly not.** "Effectively zero" (line 165), "converges to (2, 0)" — the book prints w = 1.9898, b = 0.0299 and the path plot stops short. Reason (unexplained): H = [[15,5],[5,2]] has eigenvalues 16.70 and 0.299 (κ = 55.8); at η = 0.05 the modes decay as 0.165^k and 0.985^k, and 0.985^200 = 0.049 — 5% of the slow-mode error remains. The most teachable fact in the lesson is invisible.
- Figure 3 is a vertical line at steps 0–3; nothing readable for the other 197 steps. `semilogy` shows two straight-line regimes.
- Figure 1 uses index axes 0–40 and line 96 spends a paragraph converting indices to (w,b). `contourf(W_mesh, B_mesh, L_matrix)` gives real axes and the path overlays with `hold` — one figure replaces 1, 2 and 4.
- "Elongation along the b axis" (112) / "the trajectory curves" (185): the valley runs along the slow eigenvector (1, −2.94), and the path is one sharp bend then a straight crawl along it (slope ≈ −3, visible in plot 4). Say eigenvector.
- Column convention y = Wx + b (21) vs row convention in 01/04. `real(mean(…))` (48–49) is an unexplained wrapper.

**3. Missing / thin.** Bigram-as-linear-layer (absent — the checklist's central item); η < 2/λ_max and the dynamical-system view (absent; exercise 1 asks for the divergence threshold with no tool — it is 0.1198); condition number / zig-zag (at η = 0.1 the fast pole is −0.67 → oscillation; at 0.12 it is −1.004 → divergence); batch vs full gradient (one sentence → 18); why MSE here vs CE (03 bridge); an LR experiment (table only).

**4. Reframing (exact; the natural home of the controls thread).**
- GD on a quadratic is the LTI error system e_{k+1} = (I − ηH)e_k: poles 1 − ηλ_i, stable iff η < 2/λ_max, fastest when balanced, κ = spread of time constants. `eig(H)`; print poles for η = 0.05/0.10/0.12/0.20. Optional: `ss`/`step` from the controls toolbox on the error system — exact here.
- GD as a feedback loop: mermaid block diagram (θ integrator with gain η ← −∇L ← model ← data); foreshadow momentum (16) as a second-order loop and Adam as per-axis gain (preconditioning).
- Anisotropy = eigendecomposition of H; draw the eigenvectors at the minimum with `quiver`.
- Bigram-as-linear-layer: W ∈ ℝ^{|V|×|V|}, logits = one-hot·W, softmax rows, CE; GD converges to W_ij = log P_ij + c_i — the promised "look-up vs learn" equivalence in ~25 lines reusing 05's C.

**5. 0.3.7.** `contourf` + overlaid path; `semilogy` loss; `<!-- grid: 3 -->` loss curves for η = 0.05/0.10/0.12; `saveanim` of the descent (20 frames); `eig`, `ss`/`step`; mermaid feedback diagram; `quiver` eigenvectors.

**6. Defects.** Numbers match book, but "Final b ≈ 0.0" for 0.0299 and "effectively zero" mislead given the slow mode. Orphan tracked `book/plots/06-…/plot-3-33b02aca.svg`. Line 27 equates embedding lookup with a linear layer using W = Eᵀ implicitly — say which convention.

**7. Severity: major rewrite** — the narrative promise is unmet, the central controls content (stability, conditioning) is absent while its evidence sits unexplained on the page, and three of four figures need re-forming.

---

## Cross-lesson summary (01–06)

**Notation drift.**
- T = temperature (02) vs sequence length (01, 03, 04, 05). Rename temperature τ or length N.
- Row vs column vectors: 01 rows of X; 04 flips within one section; 06 column Wx + b; later (nanoGPT-style) lessons are row-major. Standardise on rows: h = xE, y = xW + b, W ∈ ℝ^{d_in×d_out}.
- Log base: bits in 01/02, nats in 03/05/06, with 2^H and e^L both called perplexity. Rule: code in nats, report bits, always label.
- Indexing: 1-based prose/tables vs 0-based plot axes (02 lines, 04/06 imagesc) and a 0-based comment in `embedding_matrix.rlab`.
- Distribution symbols f, p, p̂/y, P_ij are tolerable, but a one-table notation reference in 01 (or `_setup.md`) would end the drift.

**Repeated pain points.**
- Editor meta-sentence "This section is pure reference… pairs ### Theory with ### Example" on student pages in 01/02/03/04.
- Figures: PMFs as lines; 3×1 stacks where a grid fits; surf renders with no axis labels; autoscaled colourbars; legend placeholders "bar"/"scatter"; heatmap labels on cell edges.
- Stale 0.3.6 workarounds (04 shift, 06 index-axis paragraph, `real()`), orphan committed plots (03, 06), truncated long prints (05).
- "Connection to Information Theory" exists in 01/02/03/05 and is the course's strongest spine; 04 and 06 have none — precisely where the EE (correlation/codebook) and controls (LTI error dynamics) sections belong. Add a parallel "Connection to Signals & Control" H2 convention.

**Three highest-leverage changes.**
1. **Lesson 06:** add the bigram-as-linear-layer example and the discrete-time stability analysis (poles 1 − ηλ_i, η < 2/λ_max, κ), with `contourf` + path overlay and `semilogy` loss.
2. **Lesson 05:** stationary distribution via `eig(P')`, the period-2 pole, smoothing → ergodic, entropy rate Σπ_iH_i, decoded generated text, mermaid state graph.
3. **Hygiene pass over 01–06** (T/τ, row convention, bits/nats, 1-based axes, remove boilerplate and stale workarounds) plus the **Lesson 03 gradient fix (p − y)** — the one factual-pedagogy error in the range.
