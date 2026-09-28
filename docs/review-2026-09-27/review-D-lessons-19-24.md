# Review D — Lessons 19–24 (read-only)

Method: read every `notebooks/NN-*.md` in full; diffed `book/` against source to see captured output; skimmed all 16 `.rlab` scripts; ran scratch copies of every script under rustlab 0.3.7 (repo untouched) to check quoted numbers. All scripts finish in < 1 s (capstone 0.84 s; each lesson-24 script 0.3 s). Line numbers refer to `notebooks/`.

---

## Lesson 19 — Byte-Pair Encoding

**1. Covers.** Char/word/subword motivation table; merge-step pseudocode; three merges on `abracadabra`×3 with tie-break discussion; the five learned merges applied to six fragments, bucketed into a 3-bar chart; a PMI paragraph; vocab-size vs parameter cost. Figures: 1 (3-bar chart). Scripts add a merge-count bar and two length charts.

**2. Clarity (ranked).**
- "Token-Length Distribution" (L121–187) promises a long-tail histogram; the figure is three bars over six hand-picked fragments, and L131/L187 describe a distribution the reader cannot see.
- The IT paragraph (L193–197) is one 200-word block with three hedges, ending in "approaches the corpus's character-level entropy from above". A tokeniser is not a code until you say how tokens are coded, and this corpus is periodic (entropy rate ≈ 0), so the sentence is not well-defined. Replace prose with a computed table (see §4).
- L11 uses PMI before defining $p_{ab}$; say "pair frequency over the $T-1$ adjacent positions".
- The hidden block (L56–67) prints, so the book shows "Initial seq length: 35" with no visible code.
- `apply_merge` (L143–155) is `bpe_step`'s rewrite loop copy-pasted; factor once.
- Column-major tie-break (L119) is an implementation artefact promoted to prose; reads as a bug.
- Greedy is never justified; one sentence (optimal dictionary selection is NP-hard; greedy pair merging is the tractable approximation) would answer Exercise 2.

**3. Missing / thin.**
- Byte-level BPE (256-byte base, GPT-2 byte→unicode mapping), special tokens (`<|endoftext|>`, BOS/EOS/pad), pre-tokenisation (regex split; why merges never cross whitespace) — absent (grep confirms). Lesson 21 later uses "EOS" without it having been introduced anywhere.
- Compression ratio / tokens-per-char only in scripts (`bpe_train` prints 3.18; `bpe_apply` 3.67/3.29/1.57), not in the notebook.
- Vocab-size vs sequence-length curve exists only as Exercise 4; it is the central design trade-off and should be a figure.
- Huffman/LZ comparison absent; lesson 01 L154–161 already introduces the source-coding bound, so the hook is waiting.

**4. IT reframing.**
- Exact: BPE *is* a compression algorithm — Gage 1994 ("A New Algorithm for Data Compression"), identical to Re-Pair grammar compression (Larsson–Moffat 1999); Sennrich 2016 repurposed it. Say it in paragraph one; the EE reader then places it (offline, static-dictionary, greedy) relative to Huffman (order-0 optimal) and LZ78 (adaptive dictionary).
- Exact computation to add (≈15 lines): total bits for the 35-char corpus under (i) 8-bit ASCII = 280; (ii) fixed ⌈log₂6⌉ = 3 bits → 105; (iii) order-0 bound $35\,H(\text{chars}) \approx 35 × 2.24 ≈ 78$ (Huffman ≈ 80); (iv) after $k$ merges: $N_k ⌈\log_2 |V_k|⌉$ + merge-table cost (2 ids/merge). Plot total bits vs $k$: the minimum is the MDL-optimal vocab size and *shows* the diminishing returns asserted at L199.
- Exact: tokenisation changes entropy per symbol but not total information: $H_{\text{tok}} N_{\text{tok}} ≈ H_{\text{char}} N_{\text{char}}$; bits/char is the invariant — the bridge to lesson 20's BPC.
- PMI-as-coding-gain: keep, but demote to Exercise 5 where it already lives.

**5. Rustlab 0.3.7.**
- Mermaid merge tree (r,a→ra; a,b→ab; ab,ra→abra; c,a→ca; ca,d→cad) in place of L119's prose.
- `<!-- grid: 2 -->`: bits-vs-$k$ curve beside tokens-per-word bars; `yline` for the entropy bound.
- `<!-- exercise -->`/`<!-- solution -->` for Exercise 5 with a live solution block (markdown output pending upstream fix).
- `run bpe_lib.rlab` to share `bpe_step`/`apply_merge` with `capstone.rlab` (own copy at L68–113 there).

**6. Defects.** None numerical: verified 35→29→23→17→14→11, first merge (5,1), fragments (3,1,1,2,2,1). Expected Outputs table agrees with book.

**7. Severity: moderate rewrite** — algorithm section is sound; the IT section must become a computation, and byte-level/special-token/pre-tokenisation coverage is absent.

---

## Lesson 20 — Perplexity and Evaluation

**1. Covers.** $\mathrm{PPL} = e^{\mathcal L}$, endpoints, $e^{H}$ of three reference distributions, PPL-vs-confidence sweep, "why PPL", bigram-floor sanity, model-comparison caveats, compression connection (prose), per-token distribution (prose), a 110-line `parmap` sidebar with a histogram. Figures: 2 (sweep; histogram of an *untrained* model). The PPL train/val curves exist only in `perplexity_curve.rlab`.

**2. Clarity (ranked).**
- $H(q)$ vs $H(p,q)$ conflation. `entropy_to_ppl` (L49–52) and the sweep (L65–81) compute $\exp H(q)$ of the model's own distribution; the objective (L7) and boxed equation (L27) are about cross-entropy on data. An IT reader will stop here. `perplexity_basics.rlab` §"Bridge" (PPL = $1/q_{\text{target}}$) does the right thing but is not in the notebook. State both definitions once and overlay $1/p$ on the sweep figure.
- The `parmap` sidebar (L168–280) is 35 % of the lesson and about a language feature; fold into `<!-- details -->` or an appendix.
- The histogram (L267–273) is of an untrained model; L280 then explains the interesting shape only appears when trained — the figure shows the uninteresting case. Training takes 0.1 s.
- "Floor" names four numbers in one lesson (1, 1.4148, 1.4042, and 3 as "reference").
- L129 says to normalise to bits/byte but never shows $\text{BPC} = \frac{N_{\text{tok}}}{N_{\text{char}}}\cdot\frac{\mathcal L}{\ln 2}$.
- L91 "linear-in-uncertainty": for this audience just say PPL is the geometric-mean inverse probability, i.e. the antilog of a code length.

**3. Missing / thin.**
- BPC conversion (formula + one number) — exercise only.
- Shannon's ≈1 bit/char for English: lesson 03 L165 mentions it; place the toy model on that axis.
- Arithmetic coding: asserted (L148, L154; also lesson 05 L263) never demonstrated; $\sum -\log_2 q(x_t)$ on the 20-token corpus vs $20\log_2 3 = 31.7$ bits is ten lines.
- Train/val overfitting diagnosis and held-out protocol (tokeniser fit on train only, leakage): notebook has no train/val curve; Expected Outputs L317 "val similar to train" — actual 1.470 vs 1.404.

**4. IT/EE reframing.**
- Exact: PPL $= 2^{H(p,q)}$ = antilog of the arithmetic-coding length per symbol (total overhead < 2 bits). Build the lesson on that identity; compression, BPC invariance and "lower is better" all follow.
- Exact: the per-token loss histogram is the distribution of *self-information* $-\log q$; mean = CE, std = varentropy, which sets the confidence interval of a PPL estimate (why held-out sets must be long). `std(losses)` is one line.
- Exact-ish: constant-factor-per-step improvement is a straight line on `semilogy` (L103 says so; nothing is plotted).
- Loose: train/val gap as MDL (model bits + data bits) — one sentence.

**5. Rustlab 0.3.7.**
- `run` the lesson-18 trainer (or a shared `bigram_lm.rlab`) instead of the hidden retrain (L174–189); histogram then uses trained weights.
- Bring `perplexity_curve.rlab`'s PPL curve into the notebook with `semilogy` + `yline(3)`, `yline(1.4148)`; `<!-- grid: 2 -->` loss beside PPL.
- `<!-- details: parmap -->` for the sidebar.
- Mermaid: corpus → tokeniser (fit on train) → split → per-token CE → PPL/BPC.

**6. Defects.**
- `perplexity_curve.rlab` L133–134: `hline(3.0, "dashed", ...)` passes a style where a colour is expected; rustlab 0.3.7 warns `unrecognized color 'dashed'` twice (verified). Use `yline(3.0, "gray", "uniform PPL = 3")`.
- L27 product uses $x_{<t}$ where L23 uses $x_1..x_t$ ($x_{\le t}$).
- L297 Key Takeaway "$2^{\text{bits/char}}$" — it is $2^{\text{bits/token}}$ unless tokens are characters.
- Expected Outputs match book (4.0 / 2.5611 / 1.1826).

**7. Severity: moderate rewrite** — reorder around the code-length identity, fix the $H(q)/H(p,q)$ muddle, fold the sidebar, add BPC + arithmetic coding, show the training curve.

---

## Lesson 21 — Sampling and Generation

**1. Covers.** Loop formalism; 49-line retrain of the lesson-18 model; greedy and mode collapse; four transforms with math; 10-class demo (2×2 bars, entropies); temperature demo on the 3-token model; logit controls; KV cache derivation, FLOP formulas, 4.3 GB real-system arithmetic; gallery pasted from script. Figures: 1 in notebook; 3 more in scripts.

**2. Clarity (ranked).**
- The temperature demo (L266–298) is on $P(\cdot\mid b)$ with logits (+8.39, −8.84, +8.39). Two tied logits stay tied at any $T$. Measured: $T=0.5, 1, 2$ all give (0.500, 0.000, 0.500) ($P(b|b)=9{\times}10^{-5}$ at $T=2$; even $T=5$ gives 0.492/0.016/0.492). L298 ("T=0.5 sharper… T=2.0 nearly random") is false and the book output (all three samples corpus-consistent) shows it. `generation_loop.rlab` L144's figure title "T=0.5 sharpens, T=2.0 flattens" plots three identical bar triples. Use the 10-class vector (animate $T$) or the capstone model.
- L42–90 (49 lines) is the verbatim lesson-18 trainer; hide or `run`. L168–220 (53 lines) should be split per transform (one H3 per example, per AGENTS.md).
- L373–392 prints a *hard-coded string* ("~5e-17") as if it were output while claiming "computed live so the totals cannot drift". Objective L10 says "prove"; the notebook asserts. (`kv_cache.rlab` measures 5.55e-17.)
- $T$ is temperature (L131–137) and sequence length (L353–397) in the same lesson; $d$ in the KV section is $d_{\text{head}}$.
- L309 is a 130-word bullet mixing definition, sign rule, HF note and result; split.

**3. Missing / thin.**
- Beam search: absent (grep) — needs a paragraph (MT use, degeneration on open-ended text, length normalisation).
- Entropy under each control: present for the 10-class demo, absent for the trained model where it would expose the non-effect above.
- KV cache as *state* and generation as feedback: not framed. Memory arithmetic and FLOPs ✓ (verified 3564/708/5.03; per-step ratio exactly $t$).
- EOS is used but never defined in the course.

**4. Controls/IT reframing.**
- Exact: greedy decoding of a Markov-$k$ model is an autonomous map on a finite state space ($|V|^k$ states), so every orbit is eventually periodic with period ≤ $|V|^k$. For the bigram (3 states) `abab…` is not a failure mode, it is the only possibility. One paragraph + a 3-node transition diagram of $\arg\max P(\cdot|x)$; it also explains why the capstone (state = whole prefix) does not collapse.
- Exact: sampling makes it a Markov chain; $z/T$ is a Boltzmann tilt with $T$ as temperature (lesson 02's language); entropy-vs-$T$ curve = "noise power vs knob".
- Exact: the loop is closed-loop with unit delay $x_{t+1} = g(f_\theta(x_{1:t}), \varepsilon_t)$; the KV cache is the plant state whose dimension grows with $t$ — the memory problem GQA/sliding-window/SSMs attack. Block diagram: prompt → model → logits → [penalty, bias, $T$, top-k/p] → sampler ← noise → $z^{-1}$ → append → back. Logit controls are a forward-path compensator.
- Loose (state once): repetition penalty ≈ negative feedback from an FIR window of past outputs; logit bias ≈ feed-forward offset; ban ≈ hard constraint.
- Exact: top-p = smallest set of mass ≥ $p$ — the typical-set idea.

**5. Rustlab 0.3.7.**
- Mermaid closed-loop diagram in "The Autoregressive Generation Loop".
- Animation: `frame()` over $T\in[0.2,5]$ of the 10-class bars with $H$ in the title → `saveanim("temperature_sweep.gif", 8)`; same for top-p.
- `run kv_cache.rlab` to replace the fake print; `<!-- grid: 2 -->` per-step + cumulative FLOPs; `loglog` for Exercise 4.
- `run` the trainer (also dedups the three lesson-21 scripts, each of which retrains it).
- `<!-- details -->` around the 53-line transforms block.

**6. Defects.** L298 prose vs numbers; `generation_loop.rlab` L144 title; L390–391 fabricated output; L476 says the capstone uses the KV cache — `capstone.rlab` L469–471 and lesson 22 L186 say it does not. Expected Outputs otherwise ✓ against runs (H ordering 1.020 < 1.040 ✓; top-p support 5 ✓; gallery strings identical).

**7. Severity: moderate rewrite** — math/code correct; flagship temperature demo broken, beam search missing, feedback framing needed.

---

## Lesson 22 — Putting It All Together

**1. Covers.** 92 chars → 8 BPE merges → 300-param single-block transformer, 600 AdamW steps with full backprop → gallery at 4 checkpoints → 4 diagnostics (script only). One live block (bigram floor). **Zero figures in the rendered notebook**; every other number is pasted text. `capstone.rlab` is 634 lines and runs in 0.84 s.

**2. Clarity (ranked).**
- No figure. §4 (L164–173) describes a 4-panel plot the reader never sees; Exercise 1 asks the reader to inspect the attention matrix the notebook never shows. The script runs in < 1 s: `run capstone.rlab` and plot.
- Maintainer changelog in student prose: L61 ("The OLD lesson 22 capstone…"), L134 NOTE ("the often-quoted 1.51 … not $\sqrt2$ either"), L136 ("22-bigram"), and `capstone.rlab` L155–161. A first-time reader has never seen 1.51 or 1.414 here. Delete.
- L61 "(computed live in the block above)" — the block is *below* (L91–116).
- Spoiler structure: gallery + all conclusions before Learning Objectives (L5–18), repeated in §3 and Key Takeaways.
- Reading order: the capstone depends on lesson 24's backward pass (L5, L22, L53, L55, L122). Either renumber (22 backprop → 23 capstone → 24 variants → 25 fine-tuning) or make 22 forward-only.
- L59 "small because rustlab is an interpreter" is no longer true (0.84 s); $d_{\text{model}}=16$ and a few hundred tokens would cost nothing.
- The headline "attention resolves the `at ` ambiguity" is admitted unproven in the same paragraph (L18, Exercise 3: PE alone suffices). The ablation (attention off / PE off / both) is a 3 s experiment; run it in the lesson as a 3-row table.
- Merge table (L74–83) is pasted in a format different from the script's.

**3. Missing.** Loss/PPL/grad-norm figures; checkpoint attention maps; any *design* discussion (why $d=4$, 8 merges, 600 steps, LR 0.05 — a params-vs-tokens budget line: 300 params for 31 pairs); KV cache (promised by lesson 21); any generalisation check (the train/val removal at L136 is argued, but "overfitting" then never appears in the capstone). References to prior lessons ✓ (table L31–53; lesson 07 missing). Bigram floor ✓ (live, 1.4723 verified).

**4. EE/IT reframing.**
- Exact: loss drops 4.5 decades; linear axes hide everything after step 50. `semilogy` loss and grad-norm, and name the phases (warm-up transient, exponential convergence, round-off floor) — the closed-loop step response of the optimiser.
- Exact: PPL 1.00008 ↔ 0.00012 bits/token ↔ ≈0.004 bits for the whole corpus; the model *is* the corpus. MDL framing: 300 params × 64 bits ≫ 32 tokens × log₂18 = 133 bits — a lossless but grossly over-parameterised code. That is the honest meaning of "memorised" and motivates held-out evaluation.
- Exact: $H(\text{next}\mid\text{cur}) = 0.387$ nats, $H(\text{next}\mid\text{cur},\text{prev}) = 0$; attention is the mechanism giving access to $I(\text{next};\text{prev}\mid\text{cur})$. Compute both live (10 lines) — the bigram-floor derivation in IT terms.
- Loose: the softmax row at an `at ` position is a time-varying FIR tap vector over the prefix; show the heatmap.

**5. Rustlab 0.3.7.**
- `run capstone.rlab`, then live plots; mention `cache enable` as the scaling pattern.
- Signed `heatmap` of $A$ (32×32) and of $W_U E^\top$ (token→token logit map).
- `<!-- grid: 4 -->` diagnostics with `semilogy`.
- Animation: attention map at steps 0/25/50/100/300 → `saveanim("capstone_attention.gif", 2)` with the decoded greedy sample as caption.
- Mermaid pipeline with lesson numbers on each box (replaces the table's "map" role).
- Shared library: L195–333 of `capstone.rlab` is lesson 24's forward/backward verbatim (144 identical non-comment lines); `run transformer_lib.rlab`. Revert `acts`→`cache` (TODO at L230–231).

**6. Defects.** L61 ordering; L136 "22-bigram"; lesson 21 L476 vs L186 (KV cache); L59 stale premise; Key Takeaway L193 "Lessons 01–24" (23 unused). Numbers all verified against a fresh run (300 params; L 2.8724 → 8.17e-5; PPL 1.00008; gallery identical).

**7. Severity: major rewrite** — content correct, but no figures, no design discussion, leaked edit history, forward dependency on 24, and the headline claim left unproven by its own admission.

---

## Lesson 23 — Modern Architectural Variants

**1. Covers.** RoPE ($\theta_k$, 2×2 rotation, relative-position identity, numeric check), RMSNorm (formula, op count, equal/diverge demo), SwiGLU (formula, 8/3 parity, forward), GQA (cache formula, LLaMA-70B numbers, group map + one head), composed stack. **Zero figures in the notebook**; 4 in scripts (RoPE diagonal heatmap, RMSNorm rows, SiLU/GELU/ReLU, cache vs $H_{kv}$).

**2. Clarity (ranked).**
- The RoPE constant-diagonal heatmap (`rope.rlab` §7) is the most convincing artefact and is script-only; likewise the cache-vs-$H_{kv}$ curve.
- L44–46 jumps to $q^\top R(m)^\top R(n)k = q^\top R(n-m)k$ without saying $R$ is block-diagonal 2×2 and that $R(m)^\top R(n) = R(n-m)$ relies on 2-D rotations commuting — the one step a linear-algebra reader needs.
- L48 ("never saw position 9000 … same offsets") is contradicted by L119 ("does not extrapolate zero-shot"). Resolve: offsets *within* the trained range are reused; beyond it they are new, and the slow pairs never completed a cycle (Exercise 1 computes this: wavelength $2\pi\cdot10^4 \gg 4096$).
- L142's "4d+5 vs 2d+3 ops" is an invented op model; say "one reduction, no subtraction, no bias" and quote the measured 1.4×.
- L222 "data-dependent on/off switch" is folklore; Shazeer 2020 explicitly offers no explanation. The real content is parity arithmetic.
- L373 "any modern open LLM uses all four" contradicts L3. Expected Outputs L395 note about 256-multiples is about LLaMA, but the row is the $d=8$ toy.

**3. Missing (checklist).** Sliding-window attention, ALiBi, MoE, FlashAttention (IO-aware), inference quantisation, speculative decoding — absent except as names at L413–418. Also: attention sinks, RoPE base-scaling formula, weight tying (one clause L362). Sliding-window is a 3-line change to the mask at L320–324; quantisation is now buildable (below). KV budget ✓ (20 GB / 2.5 GB / ~310 MB verified).

**4. EE reframing.**
- Exact and verified: RoPE is complex modulation. $z_k = x_{2k-1} + j x_{2k}$; $\mathrm{RoPE}(x,m)_k = z_k e^{jm\theta_k}$; score $=\Re\sum_k q_k \bar k_k e^{j(m-n)\theta_k}$. In rustlab 0.3.7, `qz .* exp(1i*m*theta)` reproduces `rope_apply`'s dot product to the last digit (−0.39713908851546376 both ways = $M(4,7)$ in the script's heatmap). Relative position is the phase difference of two carriers at the same frequency (mixer/heterodyne identity); $\theta_k$ are carriers spanning five decades; the heatmap diagonal is $\sum_k |q_k||k_k|\cos((m-n)\theta_k+\phi_k)$. The 13-line loop becomes one line and the DSP reader owns it.
- Exact: RMSNorm = AGC without DC removal; LayerNorm = DC-block + AGC. The shifted-input demo (L189–192) is a DC-offset test — say so, `stem` the rows.
- Exact: SwiGLU is a multiplier (mixer) between two linear projections — a bilinear form gated by SiLU; the Hadamard product gives the FFN cheap second-order terms. Better "why" than a switch.
- Exact: decode is bandwidth-bound — each step reads the whole cache once, so tok/s ≤ HBM BW / cache bytes: 20 GB at 3.3 TB/s → 6 tok/s vs 2.5 GB → 50 tok/s. That is GQA's operational reason.
- Rate–distortion for quantisation (measured with `quantize`/`qfmt`/`snr` on $\mathcal N(0,0.3^2)$ weights): int8 Q1.7 → 36.3 dB, Q3.5 → 24.5 dB, Q5.3 → 12.4 dB, int4 Q2.2 → 6.4 dB — exactly 6.02 dB/bit, the ADC rule the reader knows; then KL$(p\|p_q)$ of the softmax under quantised logits is the distortion that matters.

**5. Rustlab 0.3.7.**
- Complex-phasor RoPE (`exp(1i*m*theta)`) as primary; real 2×2 form as the storage note.
- Signed `heatmap` of $M$ (range −0.92…1.09) in the notebook; `rope.rlab` L125–128 shift workaround is obsolete.
- `polar` of $q_k e^{jm\theta_k}$ for $m=0..7$ (fast and slow spirals).
- `<!-- grid: 2 -->` RoPE heatmap + cache-vs-$H_{kv}$; `stem` for RMSNorm rows.
- New "Quantisation" section with `qfmt/quantize/snr`, `int8`, `semilogy` + `yline` for 6 dB/bit.
- Sliding-window: banded mask + heatmap. Mermaid for the composed block (L354–363 ASCII) with the four swaps highlighted.

**6. Defects.** L48 vs L119; L373 vs L3; Expected Outputs L395 note; stale comment `rope.rlab` L125. Numbers ✓ ($\theta=[1,0.01]$; dots equal to 1e-15; 552/504; GQA table).

**7. Severity: moderate rewrite** — four correct deltas, but figure-less, half the canonical list missing, and RoPE should be rebuilt on the phasor form.

---

## Lesson 24 — Full Backprop and Fine-Tuning

**1. Covers.** `abb` corpus and bigram floor; backward theory (softmax row, LayerNorm, residual, scatter-add); gradient-check numbers (pasted); training result (pasted); SFT with mask + catastrophic forgetting (pasted); DPO loss, properties, gradient, $\ln 2$ anchor (pasted); pipeline summary. **No live code and zero figures in the notebook**; three script figures. Scripts 462/457/498 lines, each < 0.3 s.

**2. Clarity (ranked).**
- The lesson that "opens the black box" has no executable block; the DPO-vs-forgetting side-by-side plot (the money figure) is never shown.
- "Derive the full backward pass" (L13) delivers 2 of ~7 pieces. Attention backward ($dQ, dK, dV$, $dS$ with $1/\sqrt d$ and mask), GELU′, CE→logits ($p - e_y$), $W_O$, and the scatter-add are one line or absent; `backward` (`full_backprop.rlab` L183–289, ~100 lines) is the only source.
- DPO is stated, not derived (L169–175): no Bradley–Terry, no KL-regularised objective, no $\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$, no $Z$ cancellation; L179 asserts the result and moves on.
- SFT "sketch" (L143–157) is pseudo-rustlab in a `text` fence; make it real.
- The PE confound (L42, L110) is honest but the fix (ablate PE, 0.3 s) belongs in the lesson, not Exercise 2.
- L102 "~0.63 (random init over a 2-token vocab)": say why it is below $\ln 2 = 0.693$.

**3. Missing / thin.**
- DPO derivation; $\beta$ as inverse temperature; why the reference is *frozen* (it defines the KL anchor).
- RLHF/PPO context: two sentences (L179–180, L257); needs a reward-model → PPO → KL-penalty diagram.
- LoRA: one clause (L137); a rank-1 $W_U + BA$ demo is ten lines and shows parameter count and reduced forgetting.
- **Likelihood displacement.** The script's own output after DPO: lp(chosen) = −19.2, −16.4, −0.27 nats (margins +29, +32, +47). On two of three prompts the policy prefers `aa` to `bb` *relatively* while assigning it $e^{-19}$. This is the documented DPO pathology (chosen log-prob falls; Pal et al. 2024, Razin et al. 2024); L193 "the policy prefers chosen over rejected" hides it. Print lp(chosen) beside the margin and discuss.
- Fine-tuning LR ("smaller LR", L213) never justified.
- Verified ✓: gradient check 2.2e-10 / 6.2e-11; forgetting 2.0e-5 → 2.837; DPO initial $\ln 2$.

**4. Controls/IT reframing.**
- Exact: $\max_\pi \mathbb E_\pi[r] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$ has solution $\pi^* = \pi_{\text{ref}}\,e^{r/\beta}/Z$ — a Gibbs tilt of the prior with temperature $\beta$ (softmax with temperature, lesson 02; minimum relative entropy, Kullback/Jaynes). Invert: $r = \beta\log(\pi^*/\pi_{\text{ref}}) + \beta\log Z$; Bradley–Terry $\sigma(r_w - r_l)$ cancels $Z$ → DPO. Six lines.
- Exact: the DPO weight $\beta(1-\sigma)$ is an error-driven gain that vanishes as the margin saturates — a dead-band controller; plot weight vs margin.
- Exact: forgetting as interference — $\cos\angle(\nabla\mathcal L_{abb}, \nabla\mathcal L_{\text{SFT}})$ at the pretrained point (one forward/backward each); replay and LoRA become "restrict the update subspace".
- Exact: gradient check ε trade-off (truncation $O(\varepsilon^2)$ vs cancellation $O(u/\varepsilon)$) — `loglog` rel-error vs ε from 1e-1 to 1e-9, the V-curve every numerical-methods student has drawn (L98 says it in words).

**5. Rustlab 0.3.7.**
- `run transformer_lib.rlab`: the three scripts share 141–186 identical non-comment lines with each other and 144 with the capstone.
- Live blocks: gradient check, ε-sweep `loglog`, training curve with `yline(0.4346)` on `semilogy`; SFT/DPO curves via `<!-- grid: 2 -->`.
- Mermaid: DPO wiring (prompt → $\pi_\theta$ and frozen $\pi_{\text{ref}}$ → log-probs of $y_w, y_l$ → margin → σ; dashed no-gradient edge into $\pi_{\text{ref}}$); pre-train → SFT → DPO/RLHF pipeline.
- Revert `acts`→`cache` (TODOs at `full_backprop` L124, `sft` L102, `dpo` L87).
- `<!-- exercise -->`/`<!-- solution -->` for Exercise 6 with the derivation as solution.
- Animation: $P(\cdot\mid\text{prompt})$ bars across DPO steps.

**6. Defects.** L193–194 omits lp(chosen) ≈ −19 nats; "keeps it close to its starting point" — abb loss 7.1e-9 → 8.1e-5 (say "both ≈ 0"). L234 "~1e-2 to 1e-3" — actual 0.018. L102 "below the floor by step ~50" unverifiable in the notebook. `W_U`/`Wq` vs $\mathbf W_Q$ notation. Pasted numbers otherwise ✓.

**7. Severity: major rewrite** — scripts are correct, but the notebook is a prose summary with no DPO derivation, an incomplete backward derivation, no figures, and it hides a real pathology visible in its own output.

---

## Cross-lesson summary

**Notation drift.** $T$ = temperature (21, 22) vs sequence length (21, 23, 24); $d$ vs $d_{\text{head}}$ vs $d_{\text{model}}$ (21 KV section); LM head $\mathbf W_{\text{head}}$ (19) vs $\mathbf W_U$ (22, 24); $f_\theta$ (21) vs $P_\theta$ (20, 24) vs $\pi_\theta$ (24); "floor" for 1, 1.4148, 1.4042, 1.4723, 0.4346. A one-table notation appendix transcluded with `![[_notation.md]]` fixes this everywhere.

**Repeated pain points.** (1) Figure-less notebooks: 22, 23, 24 render zero plots, 21 one, 20 two (neither is the training curve); the good figures live in `.rlab` files only. (2) Pasted `text` outputs (21 gallery, all of 22 and 24) can drift silently; every script runs in < 1 s, so `run script.rlab` makes them live — the "interpreter is slow" premise (22 L59) is dead. (3) Maintainer history in student prose (22 especially). (4) Stale 0.3.6 workarounds: `acts` + TODO in four scripts, `rope.rlab` |value| shift, `"dashed"`-as-colour in `perplexity_curve.rlab`. (5) Reading order: 24 must precede 22. (6) Three demonstrations whose numbers do not support their prose (21 temperature, 22 attention-vs-PE, 24 DPO chosen log-prob).

**Duplication.** `full_backprop.rlab` shares 186/364 non-comment lines with `sft.rlab`, 141 with `dpo.rlab`, 144 with `capstone.rlab`; the three lesson-21 scripts each retrain the lesson-18 model (≈45 lines ×3) and the notebook does it a fourth time; `bpe_step` exists in 19 and 22. A `lessons/lib/` pulled with `run ../lib/x.rlab`: `transformer_lib.rlab` (`layernorm_fwd/bwd`, `gelu_grad`, `forward(ids, mask, θ)`, `backward`, `adamw_step`), `bpe_lib.rlab`, `sampling_lib.rlab` (categorical, $T$/top-k/top-p, penalty, `generate`), `bigram_lm.rlab` (trained lesson-18 $E, W$). Packing parameters in a struct removes the 17-argument signatures. Net: ~700 lines removed; 22/24 scripts become ~150-line drivers.

**Three highest-leverage changes.**
1. Make 22/23/24 executable with figures (`run` + signed heatmaps + `grid` + `semilogy`): show the attention map, the RoPE diagonals, and the DPO-vs-forgetting plot. Largest clarity gain per hour.
2. Add the derivations the target audience will demand and that are currently asserted: BPE-as-source-coding bits curve (19); CE = arithmetic-coding length + BPC normalisation (20); KL-regularised → Gibbs → Bradley–Terry → DPO (24); RoPE as phasor (23). Each is 6–15 lines and lands on ground the reader already owns.
3. Repair the three broken demonstrations (tied-logit temperature demo, unproven attention claim, hidden likelihood displacement) and reorder so full backprop precedes the capstone.
