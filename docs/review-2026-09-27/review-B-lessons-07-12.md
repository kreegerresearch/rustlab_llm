# Review B — Lessons 07–12 (attention core, PE, FFN, LN/residuals)

Scope: `notebooks/07…12`, skimmed `book/` renders and `lessons/*/*.rlab`. All Expected-Output tables were checked against `book/` output; numerical claims I doubted were re-run under rustlab 0.3.7 (noted inline). Line numbers refer to the `notebooks/` sources.

---

## Lesson 07 — Context and Naive Averaging

**1. Covers now.** Bigram failure on `bank`; prefix average; lower-triangular `W` (loop, vectorised, matmul agreement); one IT paragraph on mutual information. Figures: 4 (bar of P(next|bank); `imagesc` of W, X, X̄ as three separate full-width figures).

**2. Clarity problems (ranked).**
1. The hand-crafted `X` (L102–108: rows 5–6 are sums of earlier one-hots) is never read back. The X̄ heatmap is described only as "smoother" (L179); the one concrete check — row 3 = [1/3,1/3,1/3,0] — exists only in `prefix_averaging.rlab` L81–86, not in the notebook.
2. The "why" is missing: averaging is a low-pass operation; the reader is told "smoother" without being told why smoothing is the right first idea and what it destroys (order and salience).
3. Vectorised-W section (L120–132) is twice as long as the idea; `cumsum(eye)` is explained twice (L122 and L132). Fine for a MATLAB audience, but it lands where the reader expects motivation.
4. IT paragraph (L187): target is `X_t` given `X_{1..t-1}`; every later lesson uses `X_{t+1} | X_{1..t}`. The chain-rule term `I(X_t;X_i|X_{<i})` is correct but the next sentence drops the conditioning, and the running example is muddled ("it" refers to `mat`, but the token whose prediction benefits is the one *after* `it`, i.e. `was`, not `soft`).
5. L118 "W is shown below" — the next block is the vectorised construction, the heatmap comes two blocks later.
6. `imagesc` used for positions×positions and positions×dims matrices where AGENTS.md prescribes labelled `heatmap`; stale TODO at L167.

**3. Missing / thin.** The EE hook is entirely absent: nothing says W is a filter, no impulse response, no frequency response. EMA/IIR exists only as exercise 2 and is not connected to IIR. O(T²d) cost is exercise 4 only. The formal statement of Markov-1 failure as `H(X_{t+1}|X_t) ≥ H(X_{t+1}|X_{1..t})` is missing although Lesson 05 L257 already has the entropy chain rule. Not stated: prefix averaging is still a bag of words (sets up 10).

**4. EE / IT reframing.**
- **Exact:** `X̄ = W X` is a causal, linear, *time-varying* FIR filter along the token axis; row t of W is the impulse response observed at output time t (t taps of 1/t) — the growing-window moving average. Figure: `<!-- grid: 3 -->` of `stem(W(2,:))`, `stem(W(4,:))`, `stem(W(6,:))`. Add `freqz` of a fixed-length box filter to show the sinc low-pass — that *is* "smoother".
- **Exact:** EMA (exercise 2) is a first-order IIR `y_t = γ y_{t-1} + (1-γ) x_t`, one pole at z=γ; its W row is `(1-γ)γ^{t-i}` normalised. Promote to the body: two W rows side by side plus `freqz` of both.
- **Exact, cheap:** compute `H(X_{t+1}|X_t)` and `H(X_{t+1}|X_{1..t})` on the 2-sentence corpus (1 bit vs 0 bits) — the structural failure in bits, and it primes 08's row-entropy discussion.
- **Loose (mention only):** data-dependent weights = an adaptive filter.

**5. Rustlab 0.3.7.** `grid: 3` for X | W | X̄ (L164–177), removing TODO L167; `stem` for impulse responses; `freqz` for the box/EMA responses; `<!-- exercise -->` for ex 1–2 (solution directive pending upstream fix). No signed-heatmap need (all entries ≥ 0).

**6. Defects.** No numerical errors; Expected table matches `book/`. L187 example/notation muddle as above.

**7. Severity: moderate rewrite** — everything present is correct, but the filter framing that should be the spine of the whole 07→12 arc is missing from the lesson whose job is to introduce it.

---

## Lesson 08 — Scaled Dot-Product Attention

**1. Covers now.** Q/K/V projections and dimensions; scores; √d_k variance argument with an empirical check and a saturation demo; causal mask; row softmax; O = AV; parameter count independent of T; reduction to L07; IT view (row entropy, MI). Figures: 5 heatmaps (S, masked S with +1e9 shift, A; A₂, O).

**2. Clarity problems (ranked).**
1. **The "interpretable pattern" (L32–52) is never interpreted.** Re-running it: row 2 puts 0.73 on t1; row 3 puts 0.58 on t2; row 4 puts 0.43 on t1 (vs 0.25 for uniform averaging) — content-based retrieval of a token three positions back. None of this appears in the prose; the heatmap has no key.
2. **The "Full Pipeline" example (L203–283) demonstrates nothing.** With one-hot X and identity-slice W_Q, scores are ≤ 0.5 and A₂ is near-uniform (row 4 = [0.26, 0.26, 0.21, 0.26]; entropy 1.99 bits vs log₂4 = 2.00). Plot 4 is visually the L07 averaging matrix. Scale the projections or design X so token 4 retrieves token 1.
3. Symbol overload never acknowledged: L07's mixing matrix was **W**; here W is a projection and the mixing matrix is **A**. One sentence fixes it.
4. L189–199: the +1e9 shift figure and the paragraph apologising for it are obsolete under 0.3.7.
5. L122 NaN rationale is wrong for causal masks: the diagonal is always unmasked, so the row max is finite and `−∞ − max = −∞`, `exp → 0`. NaN only arises for fully-masked rows (padding). The honest reasons for −1e9 are dtype portability and padded rows.
6. L112 introduces `p(1−p)` and backprop before Lesson 15; label it a preview.
7. The mask-building loop appears at L127–135 and L248–254 (and four more times in 09). A shared `causal_mask(T)` via `![[_setup.md]]` or `run` would cut ~25 lines across two lessons.

**3. Missing / thin.** Kernel-smoother (Nadaraya–Watson) framing — absent. Associative / content-addressable memory — one clause (L30). Matched filter / correlation receiver — absent. Row entropy — discussed (L305) but never computed. Why K and V are separate projections — absent. `rank(QKᵀ) ≤ d_k` — absent (and it is what explains the d_k=2 heads in 09). Self- vs cross-attention — "self-attention" appears at L312 with no contrast. O(T²d) — exercise 5 only. Softmax gradient — deferred, acceptable.

**4. EE / IT reframing.**
- **Matched filter / correlation receiver (exact):** `s_{t,i} = q_t·k_i` is the correlation of template q_t against each stored waveform k_i; softmax is a soft-argmax detector; 1/√d_k normalises the noise floor to unit variance. Figure: for one query row, bars of scores and of softmax; **animate** the scale factor 0.1→3 with `frame()/saveanim` and put `H(A_t)` in bits in the title — uniform → one-hot in one GIF.
- **Kernel smoother (exact, three lines):** `o_t = Σ_i K(q_t,k_i) v_i / Σ_i K(q_t,k_i)` with `K = exp(q·k/√d_k)`; L07 is the box kernel. This single formula makes 07→08 one idea rather than two.
- **Associative memory (exact as description):** keys = addresses, values = contents, query = probe. Mermaid: `X → {W_Q,W_K,W_V} → S → mask → softmax → A·V → O`.
- **IT (exact):** print `H(A_t)` beside `log₂ t` for every row, and `2^H` as the "effective number of tokens attended". Also compute `rank(S)`.

**5. Rustlab 0.3.7.** Mermaid pipeline after L30; `grid: 3` for S | masked S | A with signed `heatmap` (delete L189–191 and L199); `grid: 2` for A₂ | O; `saveanim` for softmax sharpening (replace the two print dumps L94–110); `rank()`; exercise directives for ex 1 and 3.

**6. Defects.** L122 (NaN rationale). L199 self-described uninformative figure. L60 also needs q independent of k (minor). Expected table matches `book/`.

**7. Severity: moderate rewrite** — the mathematics is correct, but both worked examples fail to show what the prose claims, and the canonical framings this audience needs are absent.

---

## Lesson 09 — Multi-Head Attention

**1. Covers now.** Why >1 head; per-head projections; four hand-set heads (first / previous / self / uniform); concat + W_O (a permutation); 4d² parameter count; "why it works" with pruning caveat; IT paragraph. Figures: 6 heatmaps (4 heads, concat, O), all separate full-width.

**2. Clarity problems (ranked).**
1. **Head 2 smuggles in positional encoding** (unit-circle Q/K, L83–96) one lesson before 10 proves attention is order-blind. A reader will ask how head 2 knew position. One sentence ("we hand-write position into Q, K; Lesson 10 shows where the model gets it") is required.
2. **The W_O demonstration argues against its own claim.** L161 says W_O "mixes features across heads"; the demo (L216–239) uses a permutation, which is exactly the case where W_O mixes nothing. Show a W_O that sums h1.1 and h2.1 into d1.
3. Two-head pipeline (L163–228) is 60 lines: V1/V2 built by loops instead of `X(:,1:2)`; mask rebuilt (L200–205) although the helper exists.
4. L264 "pack per-head projections into three combined d×d matrices" is asserted, never drawn; a labelled heatmap of the packed W_Q with head slices makes the 4d² count obvious.
5. L126 says the four heads read "at a glance" but they are four separate full-width figures (stale TODO L129).
6. L41 "what gradient descent might recover" — no evidence or forward reference to a trained-head figure.
7. Boundary behaviour (row 1 of heads 1–2 attends to itself) unremarked.

**3. Missing / thin.** Heads as filter bank / subspace decomposition — only "different view". Positional vs content heads — never labelled, although head 1 is content and heads 2–3 are pure position. Induction-head preview — absent anywhere in the course (grep confirms). Why several narrow heads rather than one wide one — the real argument (one softmax = one convex combination per position; rank ≤ d_k per head; H heads = H simultaneous retrievals) is missing. Cost — same FLOPs as one wide head — absent. head_dim = d_model/H — stated.

**4. EE / IT reframing.**
- **Filter bank (exact identity):** `Concat·W_O = Σ_h A_h X W_V^h W_O^{(h)}` — MHA is a *sum* of H rank-limited time-varying filters; W_O^{(h)} are the synthesis filters. This one line answers "why W_O" and "why heads add up". Mermaid: X fanning to H heads, W_O summing.
- **DSP labels for the four heads (exact, delightful):** head 1 = pick off the first sample (DC hold); head 2 = unit delay `z⁻¹`; head 3 = identity; head 4 = growing box filter from L07. Attention can realise any causal LTV filter *and* choose it per input.
- **Rank (exact):** `rank(Q2*K2') = 2`; with d_k = 2 a head's score matrix lives in a 2-dimensional family.
- **IT (concrete):** replace the qualitative L278 paragraph with a table of per-row entropies for the four heads (≈0 / ≈0.5 / ≈0.5 / log₂ t bits).

**5. Rustlab 0.3.7.** `grid: 4` for the four heads (L128–149) — the single largest visual win in the range; `grid: 2` concat | O; mermaid fan-out/fan-in; signed heatmap for a mixing W_O; exercise directives.

**6. Defects.** L292 script table says "2×2 attention-matrix grid" but the script emits four SVGs. W_O demo vs L161 claim. Numbers check out (O_concat row 4 = [0.0017, 3e-6, 0.25, 0.25] matches book).

**7. Severity: moderate rewrite** — code is right; the W_O pedagogy is backwards and the DSP/filter-bank labels are the entire point for this audience.

---

## Lesson 10 — Positional Encoding

**1. Covers now.** Permutation-equivariance proof plus numeric check; sinusoidal formula; PE matrix (loop and vectorised); heatmap; rotation identity; similarity-vs-offset check; token + PE; fixed vs learned table; IT paragraph. Figures: 2 (PE heatmap with (PE+1)/2 rescale; similarity curve).

**2. Clarity problems (ranked).**
1. **Prose contradicts the plot.** L186 says the curve "decays through several oscillations, then settles" and has "unit-ish self-similarity". The book plot goes 16 → 8 (k=16) → 10.5 (k=19) → 6 (k=28) → 9.6 (k=33): no settling; and self-similarity is 16 = d/2 (the Expected table says so).
2. The closed form is never stated: `PE_t·PE_{t+k} = Σ_j cos(ω_j k)`. One line makes the curve readable (16 cosines, ~11 of which are ≈1 for k ≤ 40, hence the floor near 11).
3. L27 is muddled ("breaks full permutation invariance … leaves relative order untouched"). Precise: with the causal mask, output row t is a permutation-*invariant* function of {x₁..x_t}, so the next-token prediction is order-blind. Note also the demo `attn` (L45–53) has no mask, right after three lessons of causal attention.
4. L232 says "derived below" then "derived above" for the same property.
5. L153 overclaims: the rotation property makes relative offsets *linearly decodable*; it does not make sinusoidal PE extrapolate to unseen lengths (it is known not to).
6. L242 "minimum sufficient encoding" — an IT-literate reader will object: position needs log₂T bits; PE spends d reals. It is a redundant, multi-resolution code.
7. L81 and L192 duplicate the "embedding ≈0.1 vs unit PE" paragraph. L132–136 rescale hack obsolete.

**3. Missing / thin.** Geometric progression stated (L75) but never plotted. Oscillator-bank / Fourier-features / binary-counter analogy — absent. Resolution vs unambiguous range (aliasing) and hence **why 10000** — absent. Add vs concatenate — one sentence, no argument. Relative-offset decodability — derived but not demonstrated (fit a linear map PE_t→PE_{t+3}, show zero residual). Cross terms `(e+p)·(e'+p')` in the scores — absent. RoPE — one sentence (23 covers it later; a phasor preview here would be cheap).

**4. EE / IT reframing.**
- **Phasor bank (exact):** each pair is `e^{jω_k t}`; translation is multiplication by `e^{jω_k δ}`. With native complex numbers the rotation section is one complex multiply; `polar()` of three pairs at t and t+δ is the figure. RoPE becomes "rotate q and k instead of adding to x" — one line.
- **Autocorrelation (exact):** the similarity curve is the autocorrelation of a 16-tone signal; `stem` the 16 frequencies on a log axis — the geometric spacing is constant-Q coverage from 1 token to 2π·10⁴.
- **Aliasing / unambiguous range (exact, answers "why 10000"):** a single clock ω is unambiguous for k < 2π/ω; the slowest clock sets max range (≈63k tokens), the fastest sets resolution (~1 token). Computation: drop the slow pairs and show `PE_t ≈ PE_{t+63}`.
- **Controls (exact, one line):** `PE_{t+1} = Φ PE_t` with Φ block-diagonal rotations — position is the time index of an autonomous LTI oscillator bank (`expm` of skew blocks).
- **IT (honest):** replace L242 with "a redundant code trading bits for linear decodability", not "minimal".

**5. Rustlab 0.3.7.** Signed `imagesc(PE)` (drop L132–139 rescale); `grid: 2` PE | wavelength `semilogy`; complex + `polar` for rotation; `stem` of frequencies; `fft` of one column to show a single spectral line; exercise directives.

**6. Defects.** L186 (prose ≠ plot; "unit-ish" ≠ 16); L232 below/above; L153 overclaim; L242 "minimal"; L27 muddle. Expected table matches `book/`.

**7. Severity: moderate rewrite** — derivation is right, the single analysis plot is mis-described, and every EE hook the owner listed is absent.

---

## Lesson 11 — Feed-Forward Block

**1. Covers now.** FFN equation; position-wise independence check; ReLU vs GELU with finite-difference derivative; pre/post histograms; 4× and 2:1 parameter ratio; IT paragraph. Figures: 3 (activation overlay; two 80-sample histograms).

**2. Clarity problems (ranked).**
1. **Notation flip.** Math L23–28 is column-convention (`W₁ ∈ ℝ^{d_ff×d_model}`, `W₁x`); code L49 is `W1 = randn(d_model, d_ff)` with `H*W1` — the code's `W1` is the math's `W₁ᵀ`, never said. Lesson 08 explicitly established the row convention.
2. **L199 IT paragraph will not survive this audience.** For continuous variables `I(post;pre)` is infinite wherever the map is locally injective, and GELU is *not* injective (minimum near x≈−0.75; `gelu(−0.5) ≈ gelu(−1.05)`). The correct statement is about the Jacobian (gradient signal), not MI.
3. L131 "the entire reason GELU has displaced ReLU" — overclaim; the honest answer is empirical.
4. L201 "information bottleneck the wrong way around" concedes it is not a bottleneck; delete.
5. Histograms of 80 samples (L139–166) cannot show the "thin left tail" described; use ≥1000 draws.
6. Derivatives are computed (L113–128) but never plotted; `outer(ones_T, b1)` with zero biases is noise.

**3. Missing / thin.** **FFN as key–value memory** (rows of W₁ = keys matched against x, columns of W₂ = values, GELU = sparse gate; d_ff = number of slots) — absent, and it is the canonical answer to "what does the FFN do". "≈2/3 of block parameters" — implied by the 2:1 ratio but never said. Universal approximation — one clause. He init — one code comment.

**4. EE / Controls / IT reframing.**
- **Wiener–Hammerstein cascade (exact structurally):** linear → static nonlinearity → linear, memoryless along the token axis; attention is the dynamic (time-mixing) stage. A transformer block is the classic block-oriented alternation of LTV mixing and static nonlinearity. Mermaid `L → N → L`.
- **Small-signal gain (exact):** `GELU′(0) = ½`, →1 for x≫0, →0 for x≪0 — an input-dependent gain; the derivative plot *is* the gain curve. This pays off in 12 (collapse rate is (½)^L). ReLU is a half-wave rectifier.
- **Key–value memory (exact algebra):** `FFN(x) = Σ_m σ(k_m·x + b_m) v_m`. Figure: signed heatmap of `hidden_post` (T × d_ff) — which slots fire per token.
- **IT (honest):** drop the MI claim.

**5. Rustlab 0.3.7.** `grid: 2` activations | derivatives; `grid: 2` histograms with more samples; signed heatmap of `hidden_post`; mermaid L-N-L; exercise directives for ex 1–2. `ffn_forward.rlab` L45–48 still carries the `H − min(H)` workaround.

**6. Defects.** L199 (MI claim), L131 (overclaim), notation flip. Expected table matches `book/` (0.1326 vs "≈0.133").

**7. Severity: moderate rewrite** — correct code; the weakest "why" sections in the range, and the FFN-as-memory view is absent.

---

## Lesson 12 — LayerNorm & Residual Connections

**1. Covers now.** LN formula with γ, β; manual vs builtin; per-row LN; histograms; residual `y = x + f(x)` with the Jacobian sentence; 24-layer forward-magnitude demo; Pre- vs Post-LN with an endpoint demo; IT paragraph. Figures: 2 (LN histogram subplot; `semilogy` magnitude vs depth).

**2. Clarity problems (ranked).**
1. **Collapse mechanism is stated loosely.** L122–129 and L163 attribute the decay to "GELU clipping half the signal". Measured: total ratio 4.0e-8 ≈ 2^{-24.6}; per-layer ratios 0.3–0.8 with geometric mean 0.49; `gelu′(0) = 0.5` exactly. Once the signal is small GELU is linear with gain ½ and `randn/√d` is norm-preserving on average, so the stack is a cascade with small-signal gain ½ per stage. That is the exact mechanism and the EE-native statement.
2. **L226 mischaracterises Pre-LN growth as "exactly the clean gradient highway".** Unbounded growth is a known Pre-LN drawback (later layers' relative contribution shrinks; it is why Lesson 14 L320 rescales by 1/√(2N) and why final LN exists). The code (L213–219) also overwrites `mag_pre` each iteration, so the prose's "track the running magnitude" (L201) yields one number and no curve.
3. **L28 BatchNorm argument is wrong-headed.** BN uses running statistics at inference; the real problems are variable-length/padded sequences, cross-example statistics coupling tokens, and train/inference mismatch.
4. L163 "forward preservation is a proxy for backward" is asserted; the Jacobian product is never computed though it is ~6 lines.
5. Lesson 13 L38 refers to "the residual stream picture from Lesson 12" — 12 has no such picture; there is no wiring figure at all.
6. The 0.1 factor (L154) is never tied to real init (1/√(2N) in 14).

**3. Missing / thin.** Product-of-Jacobians stability — implicit only. Vanishing-gradient demo — forward-only. Residual stream as a shared bus that sublayers read (through LN) and write (add) — absent. Forward-Euler / Neural-ODE view — absent. AGC analogy — absent. **The strongest "why normalise" is missing:** LN is what makes the unit-variance assumption behind 08's 1/√d_k hold at the attention input. RMSNorm — not mentioned here.

**4. Controls / EE reframing.**
- **Forward Euler (exact form):** `x_{l+1} = x_l + h f(x_l)`, h = 0.1 in the demo — depth is time; the no-residual stack `x_{l+1} = f(x_l)` is an autonomous map whose origin is attracting with linearised gain ½. Add a third curve with h = 1 (exercise 3 becomes a figure).
- **Linearised stability (exact, ~6 lines with `eig`/`svd`):** compare `Π(I + h J_l)` against `Π J_l` — cumulative smallest singular value per layer. The backward pass is the same product transposed; this is the checklist's Jacobian item made concrete.
- **AGC (loose, say so):** LN = DC-block + per-vector RMS normaliser, memoryless; γ, β re-insert a programmable gain/offset. Not a loop, so not AGC in the control sense.
- **Pre-LN as integrator (exact):** uncorrelated contributions give ‖x_l‖ ∝ √l; Post-LN renormalises. Plot both curves with `yline(sqrt(d))`.
- **Bus diagram (mermaid):** residual stream as a horizontal bus with attention and FFN taps — the picture Lesson 13 assumes exists.

**5. Rustlab 0.3.7.** Mermaid bus diagram at the top; Pre/Post-LN curves (`grid: 2` with the residual plot); `eig`/`svd` Jacobian product; `yline` at √d; exercise directives. The histogram `subplot` is fine.

**6. Defects.** L28 BN reasoning; L226 growth-as-virtue; L163 loose mechanism; Pre/Post demo prints endpoints only. Expected table matches `book/`.

**7. Severity: moderate rewrite** — math is fine; the controls framing is the reason this lesson exists for the new audience, and two "why" paragraphs are wrong.

---

## Cross-lesson summary

**Does 07→08→09→10 build one mental model?** 07→08→09 genuinely does: mixing matrix W → A → A_h, with explicit reductions at 08 L295 and 09 L151. That is the range's real strength. Lesson 10 breaks the thread: it proves equivariance for *unmasked* attention (`attn` has no mask) after three lessons of causal attention, and never returns to show what PE does to A (exercise 3 only). Meanwhile 09's head 2 already uses position. The reader therefore gets one coherent picture of mixing, then a side-quest whose consequence for the attention matrix is never displayed. 11 and 12 are internally sound but each is missing the framing that would connect it back (FFN as keyed memory mirroring attention's keys/values; LN as the guarantor of 08's unit-variance assumption).

**Notation drift.** Mixing matrix W (07) vs A (08+) while W becomes a projection; P = bigram probabilities (05/07) vs permutation (10); token matrix X (07–10) vs H (11–12, and 10 L192); target X_t (07) vs X_{t+1} (08–12); column-vector math vs row-major code (11); d (07) vs d_model (08+); 10 L43 silently uses d_model as d_k. The causal mask is built by hand-loop six times across 08–09.

**Repeated pain points.** Five stale "TODO: recombine" markers and two 0.3.6 shift hacks (08 L189, 10 L132); heatmap sequences split into full-width singles where a grid is meant; hand-crafted matrices whose pattern is never read back (07 X, both 08 examples, 09 W_O); IT paragraphs loose exactly where an IT-literate reader will check (07 L187, 10 L242, 11 L199); and every EE/controls hook on the checklist is absent — the lessons are currently "ML with IT garnish", not an EE course.

**Three highest-leverage changes.**
1. **Install the filter/kernel spine and carry it through.** 07 = causal LTV FIR (box; EMA = IIR) → 08 = Nadaraya–Watson kernel / correlation receiver with row entropy printed → 09 = filter bank with the identity `Concat·W_O = Σ_h A_h X W_V^h W_O^{(h)}` and DSP labels (z⁻¹, DC hold, box) → 10 = phasor bank whose autocorrelation is the similarity curve, aliasing answers "why 10000" → 12 = forward-Euler state update with a Jacobian-product stability check. One formula and one figure each.
2. **Fix the demonstrations that do not demonstrate, then interpret every hand-crafted matrix in prose:** 08's full pipeline (near-uniform A₂), 09's permutation W_O, 10's mis-described similarity curve, 12's endpoint-only Pre/Post demo.
3. **Batch the 0.3.7 cleanup:** delete the five TODOs and two shift hacks; `grid: 3`/`grid: 4` for the pipeline and head figures; one mermaid per lesson (QKV pipeline, head fan-in/out, L-N-L, residual bus); one animation (08 softmax sharpening vs scale); `![[_setup.md]]` for `causal_mask` and the attention helper.
