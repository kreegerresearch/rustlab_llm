# Lesson 22: Putting It All Together

This is the capstone. Every prior lesson built one piece — characters, softmax, embeddings, attention, the transformer block, backpropagation, AdamW, BPE, perplexity, sampling, the KV cache. This lesson puts them all in one script and watches a tiny single-block transformer **learn**.

The corpus is the periodic phrase `"the cat sat on the mat "` repeated four times. After BPE tokenisation, training the full architecture end-to-end with [Lesson 24's analytical backprop](24-full-backprop-and-fine-tuning.md), and sampling, the model reproduces the corpus exactly:

```text
step   0 greedy : ' the cthe cthe cecat at the cthe cecthe at the cthe cec '
step 100 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 300 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 600 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 600 T=0.7  : ' the cat sat on the mat the cat sat on the mat the c '
step 600 T=1.0  : ' the cat sat on the mat the cat sat on the mat the c '
step 600 K=3    : ' the cat sat on the mat the cat sat on the mat the c '
step 600 P=0.9  : ' the cat sat on the mat the cat sat on the mat the c '
```

Three things to read from that gallery. First, **training is essentially perfect**: final PPL is $\approx 1.00008$ — practically the lower bound. Second, **context beats a bigram on this corpus**: the best any context-1 (bigram) model can do here is PPL $\approx 1.47$, because after the token `"at "` the next token is genuinely ambiguous — it is `"s"`, `"o"`, or `"the c"` depending on whether this is `"cat"`, `"sat"`, or `"mat"`, and a bigram cannot tell them apart. The single-block transformer looks one token back to see which `"at "` it is and reproduces the full corpus. (On a *fixed* corpus the transformer actually has two routes to that answer — attention over the previous token, and the fixed positional embedding, which by itself makes every position distinguishable; Exercise 3 pulls them apart.) Third, **every sampling strategy converges to the same output** — once the model is this confident, temperature, top-K, and top-P all leave the argmax untouched.

## Learning Objectives

- See **every component from Lessons 01–24 composed in one script** — tokens, BPE, transformer block forward + backward, AdamW with warmup+cosine, perplexity, all four sampling strategies.
- Watch a single-block transformer **learn to reproduce a periodic corpus** that a bigram model cannot solve.
- Verify the **context-beats-bigram** claim quantitatively: the full transformer reaches $\mathrm{PPL} \approx 1.00008$, versus the optimal-bigram floor of $\approx 1.47$ on the same corpus and the same BPE tokenisation.
- Recognise the **mode-collapse → resolution** pattern: bigram greedy collapses to a 2-cycle (Lesson 21 demo); attention resolves it (this lesson).

## Background

You have built and seen run:

| Lesson | What it contributed |
|---|---|
| [[01-tokens-and-encoding]] | character → integer-id mapping |
| [[02-probability-and-softmax]] | softmax to convert logits to a next-token distribution |
| [[03-cross-entropy-loss]] | $\mathcal{L} = -\log P_\theta(x_{t+1} \mid x_{<t})$ as training objective |
| [[04-embeddings-and-similarity]] | the trainable embedding matrix $\mathbf{E}$ |
| [[05-bigram-language-model]] | the bigram baseline and CDF sampling |
| [[06-linear-layers-and-gradient-descent]] | linear layer + gradient descent |
| [[08-scaled-dot-product-attention]] | self-attention with causal mask |
| [[09-multi-head-attention]] | parallel heads, concatenate, project |
| [[10-positional-encoding]] | sinusoidal positional embeddings |
| [[11-feed-forward-block]] | position-wise FFN with GELU |
| [[12-layer-norm-and-residuals]] | LayerNorm + residual stream |
| [[13-transformer-block]] | block = MHA + FFN with Pre-LN + residuals |
| [[14-full-gpt-architecture]] | full GPT wiring + parameter count |
| [[15-backpropagation]] | chain rule through every layer |
| [[16-adamw-optimizer]] | AdamW with decoupled weight decay |
| [[17-learning-rate-scheduling]] | warmup + cosine decay |
| [[18-training-loop]] | forward / backward / optimiser / log |
| [[19-byte-pair-encoding]] | the BPE merge algorithm |
| [[20-perplexity-and-evaluation]] | PPL as evaluation metric |
| [[21-sampling-and-generation]] | autoregressive loop + sampling strategies + KV cache |
| [[24-full-backprop-and-fine-tuning]] | analytical backward through one block — what makes this end-to-end |

The capstone script `capstone.rlab` references each by section header. The forward/backward library is copied verbatim from [[24-full-backprop-and-fine-tuning]]'s `full_backprop.rlab` so the script remains self-contained.

## What This Lesson Trains End-to-End

The capstone trains the **full single-block transformer** end-to-end with analytical gradients — 300 parameters total: token embedding $\mathbf{E}$ ($18 \times 4 = 72$), Pre-LN scales and biases ($4 \times 4 = 16$), Q/K/V/O projections ($4 \times 16 = 64$), FFN $\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2$ ($32 + 8 + 32 + 4 = 76$), and LM head $\mathbf{W}_U$ ($4 \times 18 = 72$). Sinusoidal positional embeddings are fixed (not trainable). The model is small because rustlab is an interpreter; the recipe scales transparently to LLaMA dimensions.

The OLD lesson 22 capstone (prior to [[24-full-backprop-and-fine-tuning]]) trained only the embeddings and LM head — a bigram surrogate — because backprop through attention had been derived in Lesson 15 but not yet wired into a training script. On this corpus and tokenisation the best any context-1 model can reach is the optimal-bigram floor $\mathrm{PPL} \approx 1.47$ (computed live in the block above), and the old surrogate landed near there. The irreducible ambiguity is the token `"at "`: it is followed by `"s"`, `"o"`, or `"the c"` — the continuations of `"cat"`, `"sat"`, and `"mat"` — and a bigram, conditioning only on the current token, cannot tell the three apart. **Attention removes that ambiguity.** The full transformer looks one token back — each `"at "` is immediately preceded by `"the c"`, `"s"`, or `"the m"` — and assigns probability $\approx 1$ to the correct next token. The PPL drops from the $\approx 1.47$ floor to 1.00008.

> [!IMPORTANT]
> "Mini-GPT" here means **the full GPT recipe applied to a small model**. Every component — char tokens, BPE, embeddings, attention, FFN, LN, residuals, AdamW, warmup+cosine, PPL, sampling, full backprop — is from the curriculum, with no black boxes. Every choice scales transparently to a real LLM. Only the numbers change.

## Walkthrough

### 1. Tokenise the corpus

The phrase `"the cat sat on the mat "` (23 chars, trailing space) is repeated 4 times, then tokenised character-by-character into the 10-symbol base vocabulary `{ , a, c, e, h, m, n, o, s, t}`. This is [[01-tokens-and-encoding]] exactly — 92 character ids.

Then BPE ([[19-byte-pair-encoding]]) is applied for 8 merges. The merges in order:

```text
Merge 1  pair (t,  ) count = 12   new_id = 11      % "t "
Merge 2  pair (a, 11) count = 12  new_id = 12      % "at "
Merge 3  pair (e,  ) count = 8    new_id = 13      % "e "
Merge 4  pair (t, h) count = 8    new_id = 14      % "th"
Merge 5  pair (14, 13) count = 8  new_id = 15      % "the "
Merge 6  pair (n,  ) count = 4    new_id = 16      % "n "
Merge 7  pair (15, c) count = 4   new_id = 17      % "the c"
Merge 8  pair (15, m) count = 4   new_id = 18      % "the m"
```

After 8 merges the sequence is 32 tokens long (down from 92 chars), and the vocabulary contains word-fragment tokens like `"the c"` and `"the m"` — the model now operates at a subword level.

### Example — The bigram floor these tokens impose

Before we train anything, we can compute the best score any *context-1* (bigram) model could achieve on this exact 32-token sequence. That number is the bar the full transformer has to clear. The sequence below is the output of the 8 BPE merges above (`tokens` in `capstone.rlab`); ids follow the merge table — `12 = "at "`, `17 = "the c"`, `18 = "the m"`, `16 = "n "`, `9 = "s"`, `8 = "o"`.

```rustlab
% 32-token BPE sequence from capstone.rlab: phrase [17 12 9 12 8 16 18 12] x4.
tokens = [17, 12, 9, 12, 8, 16, 18, 12, 17, 12, 9, 12, 8, 16, 18, 12, ...
          17, 12, 9, 12, 8, 16, 18, 12, 17, 12, 9, 12, 8, 16, 18, 12];
vocab_bg = 18;
T_bg = length(tokens);

% Count every bigram transition count(curr, next).
counts = zeros(vocab_bg, vocab_bg);
for t = 1:(T_bg - 1)
  counts(tokens(t), tokens(t + 1)) = counts(tokens(t), tokens(t + 1)) + 1;
end

% Optimal bigram assigns P(next | curr) = count(curr, next) / row-sum.
% Its cross-entropy is -mean log P(next | curr) over all T-1 transitions.
ce_floor = 0.0;
for t = 1:(T_bg - 1)
  curr = tokens(t); nxt = tokens(t + 1);
  p_bigram = counts(curr, nxt) / sum(counts(curr, :));
  ce_floor = ce_floor - log(p_bigram);
end
ce_floor = ce_floor / (T_bg - 1);
ppl_floor = exp(ce_floor);
print("optimal-bigram CE (nats):", ce_floor);
print("optimal-bigram PPL floor:", ppl_floor);
```

Every current token predicts its successor deterministically **except token 12 (`"at "`)**, which is followed by three different tokens across the corpus: `"s"` (in `"sat"`, 4 times), `"o"` (in `"on"`, 4 times), and `"the c"` (wrapping into the next phrase, 3 times). A bigram conditions only on the current token, so it cannot see *which* `"at "` it is looking at; the best it can do is hedge with $P = (4/11,\, 4/11,\, 3/11)$. Those 11 ambiguous transitions cost $H(4/11, 4/11, 3/11) \approx 1.09$ nats each; the other 20 are free, so the mean cross-entropy is $\tfrac{11}{31} \cdot 1.09 = ${ce_floor:%.4f}$ nats, giving $\mathrm{PPL}_{\text{floor}} = ${ppl_floor:%.4f}$. **That is the number attention has to beat** — and the trained transformer reaches PPL 1.00008 by looking one token back to disambiguate the three `"at "` contexts.

### 2. Train

The capstone runs 600 AdamW steps with warmup+cosine LR schedule ([[17-learning-rate-scheduling]]) on the **full transformer forward pass** from [[14-full-gpt-architecture]] and the **analytical backward pass** from [[24-full-backprop-and-fine-tuning]]. Every parameter receives a gradient: token embedding $\mathbf{E}$, both LayerNorm scales/biases, the four attention projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$, FFN weights and biases, and the LM head $\mathbf{W}_U$.

Output:

```text
Initial L = 2.872   PPL = 17.68   (log(vocab) = 2.890 is uniform baseline)
Final   L = 0.0000817  PPL = 1.00008
```

Loss drops from ~$\log |\mathcal{V}|$ (uniform-random baseline) to **0.0000817**, equivalently PPL = **1.00008**. The model has learned to assign probability $\approx 1$ to the correct next token at every position — perfect except for floating-point noise.

> [!NOTE]
> The best a context-1 (bigram) model can do on this corpus is **not** a clean closed form. On the capstone's own 32-token BPE sequence the optimal-bigram cross-entropy is $\mathrm{CE} = ${ce_floor:%.4f}$ nats, i.e. $\mathrm{PPL}_{\text{floor}} = ${ppl_floor:%.4f}$ (the block above computes both). The often-quoted 1.51 was the *empirical* train-split PPL of the pre-rewrite embedding-only capstone — an artefact of that particular run, not a derived bound (and it is not $\exp(\tfrac{1}{8}\cdot 4\log 2) = \sqrt2 \approx 1.414$ either). Either way, attention closes the gap by looking one token back to disambiguate the three `"at "` continuations; the full transformer reaches PPL 1.00008.

There is no train/val split in this capstone — the corpus is periodic with period 8 tokens, and once positional embeddings are involved, any held-out positions are PE-out-of-distribution, not a useful overfitting signal. Lessons 18, 20, and 22-bigram demonstrate the train/val pattern at appropriate scale; here we train on every prediction position.

### 3. Generate at checkpoints

Four snapshots — at steps 0, 100, 300, 600 — capture how the model's output evolves. The prompt is token 17 (`"the c"`), which is the natural start of every phrase in the corpus.

```text
step   0 greedy : ' the cthe cthe cecat at the cthe cecthe at the cthe cec '
step 100 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 300 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 600 greedy : ' the cat sat on the mat the cat sat on the mat the c '
```

By **step 100** the model has already learned the full corpus structure. Greedy decoding now reproduces the corpus exactly — there is no mode collapse, because the model resolves the `"at "` ambiguity from context (it reads the token immediately before each `"at "`). This is the headline contrast with the old bigram surrogate: same corpus, same BPE, same training loop, attention added — and `"sat on the mat"` now appears under *greedy* decoding, not only under sampling.

Sampling strategies behave identically at step 600 because the model is so confident:

```text
step 600 T=0.7  : ' the cat sat on the mat the cat sat on the mat the c '
step 600 T=1.0  : ' the cat sat on the mat the cat sat on the mat the c '
step 600 K=3    : ' the cat sat on the mat the cat sat on the mat the c '
step 600 P=0.9  : ' the cat sat on the mat the cat sat on the mat the c '
```

The model has effectively memorised the corpus (which is the goal — the corpus is fully deterministic given context). At inference, the next-token distribution at every position is essentially one-hot, so temperature rescaling and top-K/top-P truncation are no-ops: every strategy picks the same argmax.

The intermediate step-100 output is interesting because it shows the model has learned the periodic structure but had to bootstrap through a few half-formed states (step 0 produces gibberish heavy with `"the c"` because the random LM head happens to favour that token after most prefixes).

### 4. Read the diagnostics

The capstone saves a 4-panel diagnostic figure (`capstone.svg`):

- **Loss curve** — drops from ~2.9 at step 0 to ~$10^{-4}$ by step 600.
- **PPL curve** — same shape on the natural y-scale; ends at 1.00008. The 'effective branching factor' interpretation says the model has reduced the choice from 18 possibilities to ~1.
- **Gradient norm** — drops from $\sim 10^{-1}$ to $\sim 10^{-4}$ over training. Healthy "decrease over time" from [[18-training-loop]].
- **Learning rate** — warmup + cosine envelope from [[17-learning-rate-scheduling]].

These are the four plots a production training run would also show. Reading them is the same — only the y-axis scale changes.

## What Production Adds

The capstone covers every component of an LLM **end-to-end with analytical gradients**. A production trainer adds infrastructure on top:

- **Mini-batching.** The capstone uses one full-batch gradient per step on a 32-token corpus. Real training samples millions of tokens per batch and uses gradient accumulation across micro-batches.
- **Mixed precision** (fp16 / bf16). Forward in low precision, accumulate in fp32. Cuts memory and time by ~2×.
- **Gradient clipping.** A `max_norm = 1.0` clip step ([[18-training-loop]] sidebar) prevents single-step divergence on outlier batches.
- **Dropout.** Stochastic activation zeroing during training ([[13-transformer-block]] sidebar) — small effect on small models, important at scale.
- **Distributed training.** Data-parallel and tensor-parallel splits across many GPUs; the only architectural addition is the inter-GPU all-reduce on gradients.
- **Real corpora.** Wikipedia + Common Crawl + code, hundreds of billions of tokens, tokenised once and held in a memory-mapped file.
- **Multi-layer stacks and multi-head attention.** The capstone runs one block with one head. LLaMA-2-7B runs 32 blocks with 32 heads. The forward/backward library generalises by adding outer loops; the math is identical.
- **KV cache for inference.** Generation here recomputes the full forward pass on the growing prefix every step — $O(T^3)$ total. With the KV cache from [[21-sampling-and-generation]] this becomes $O(T^2)$.
- **Modern variants.** RoPE / RMSNorm / SwiGLU / GQA from [[23-modern-architectural-variants]] — each a local swap against the baseline used here.

Every one of those is an engineering or substitution layer over the math the capstone already builds. **The reverse — a production trainer that wires up infrastructure but gets the math wrong — fails silently and is far harder to debug.** The understanding compounds the way the model's loss does.

## Key Takeaways

- The capstone composes **Lessons 01–24** into a single script that trains a single-block transformer end-to-end with analytical gradients.
- Loss drops from $\approx 2.87$ (uniform over 18 tokens) to $\approx 8 \times 10^{-5}$ — **PPL = 1.00008**, essentially the lower bound.
- **Context beats a bigram on this corpus**: the optimal-bigram floor is PPL $\approx 1.47$ (CE $= 0.3868$ nats) because `"at "` is followed by three different tokens (`"s"`, `"o"`, `"the c"`) and a bigram cannot tell them apart; the transformer resolves it by looking one token back. On a *fixed* corpus the positional embedding provides a second, independent route to the same disambiguation — see Exercise 3.
- **Greedy decoding now reproduces the corpus** — no mode collapse — because the model's distribution is essentially one-hot at every position. All sampling strategies (temperature, top-K, top-P) converge to the same output for the same reason.
- The four diagnostic plots (loss, PPL, gradient norm, LR) are exactly what a production run reports.
- Scaling to a real LLM swaps the model size, the corpus, the precision, and the training scale — not the underlying math.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `capstone.rlab` | Full end-to-end on the periodic `"the cat sat on the mat "` corpus: char-tokenise → BPE merge (Lesson 19) → full transformer forward + analytical backward (Lessons 13, 14, 15, 24) → AdamW + warmup-cosine training (Lessons 16, 17, 18) → checkpoint sampling gallery (Lesson 21) → diagnostic plots (Lessons 17, 18, 20). Self-contained — the forward/backward library is copied from [[24-full-backprop-and-fine-tuning]]. |

Run with `make lesson-22` (or `rustlab run lessons/22-putting-it-all-together/capstone.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| Char vocab size | $10$ |
| Initial char sequence length | $92$ ($23 \times 4$) |
| Number of BPE merges | $8$ |
| Final vocab size | $18$ |
| Final token sequence length | $32$ |
| Trainable parameters | $\approx 300$ (full single-block transformer at $d_{\text{model}} = 4, d_{\text{ff}} = 8$) |
| Initial loss | $\approx 2.87$ (≈ $\log 18 = 2.89$ uniform baseline) |
| Final loss | $\approx 8 \times 10^{-5}$ |
| Final PPL | $\approx 1.00008$ |
| Optimal-bigram floor on the 32-token BPE sequence | PPL $\approx 1.4723$ (CE $= 0.3868$ nats) |
| step-100 greedy from `"the c"` | `"the cat sat on the mat the cat sat on the mat the c"` (corpus reproduced) |
| step-600 greedy / T=0.7 / T=1.0 / K=3 / P=0.9 | all identical — corpus reproduced exactly |

## Exercises

1. **Why no mode collapse?** Examine the trained attention pattern $\mathbf{A}$ at a position whose current token is `"at "` (id 12) — the only token with an ambiguous successor. Which earlier position carries the disambiguating information, and is it attended to with high weight? Hint: it is the token *immediately before* `"at "` (one position back) — `"the c"`, `"s"`, or `"the m"` — which uniquely determines the continuation.
2. **Bigram floor analytically.** Using the transition counts from the block in §1, show by hand that a context-1 model must assign $P(\cdot \mid \text{"at "}) = (4/11, 4/11, 3/11)$ over $(\text{"s"}, \text{"o"}, \text{"the c"})$, that every other current token has a deterministic successor, and that the mean cross-entropy over all 31 transitions is therefore $\tfrac{11}{31}\,H(4/11, 4/11, 3/11) = 0.3868$ nats — i.e. $\mathrm{PPL}_{\text{floor}} = 1.4723$. Confirm against the printed value.
3. **What actually forces the bigram floor?** Disabling attention alone — replace `H_mid = H_in + proj` with `H_mid = H_in` and re-train — does *not* recover the bigram floor: the model still drives to PPL $\approx 1.0$. The reason is the fixed sinusoidal positional embedding, which makes all 32 positions distinct, so the FFN can memorise a position→next-token map on this fixed corpus — a route that has nothing to do with attention. To genuinely reduce the model to context-1, ablate **both** attention and the positional embedding (also set `PE = zeros(T_max, d_model)`). Now the model sees only the current token embedding, is architecturally bigram-limited, and should floor near PPL $\approx 1.47$. Extension: ablate attention *only* and watch PPL still fall toward 1 — direct evidence that on a fixed corpus positional encoding is a second, independent route to the answer.
4. **More merges.** Re-run with $n_{\text{merges}} = 20$ instead of 8. The vocabulary grows; what happens to the sequence length and the final PPL? Past how many merges does PPL stop improving?
5. **Longer corpus.** Change `n_reps` from 4 to 10. Does the loss curve shape change? With more training pairs but the same parameter count, does the model still memorise perfectly?
6. **Different prompt.** Generate from prompt = 12 (= `"at "`), which appears 12 times in the corpus. Does greedy produce a coherent continuation? Hint: position-in-corpus matters — the model uses positional embeddings, so the same input token at different positions may continue differently.

## What's next

**This is the end of the original curriculum.** You have built every component of a GPT-style language model from first principles: probability and softmax, embeddings and similarity, attention and multi-head attention, positional encoding, the transformer block, the full stack, backpropagation through every layer, AdamW, learning-rate scheduling, the training loop, BPE tokenisation, perplexity, sampling strategies, the KV cache, and now the full end-to-end training of all of them in a single script.

The natural next steps:

- **Modern architectural variants** ([[23-modern-architectural-variants]]) — RoPE, RMSNorm, SwiGLU, GQA. Drop-in swaps against the baseline used here.
- **Fine-tuning and preference optimisation** ([[24-full-backprop-and-fine-tuning]]) — SFT with prompt masking and DPO with a frozen reference policy. Uses the same forward/backward library as this capstone.
- **Scale up.** Move to a real corpus (TinyShakespeare, Wikipedia, code); increase $d_{\text{model}}$, $N$, and $|\mathcal{V}|$. The infrastructure layer is in [What Production Adds](#what-production-adds).
- **Read [nanoGPT](https://github.com/karpathy/nanoGPT).** The 300 lines of Python in `nanoGPT/model.py` are exactly the architecture in [[14-full-gpt-architecture]], written in PyTorch. You now know what every line means.

The math you built here is the math every modern language model runs on — the rest is engineering.
