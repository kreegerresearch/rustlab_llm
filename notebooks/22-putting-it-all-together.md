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

Three things to read from that gallery. First, **training is essentially perfect**: final PPL is $\approx 1.00008$ — practically the lower bound. Second, **attention beats bigram by construction**: the original Lesson 22 used an embedding-only bigram model and floored at PPL $\approx 1.5$ with greedy collapsing to `"the cat the cat"` because it could not tell the two `"the "` contexts apart. The single-block transformer here uses attention to look at *which* `"the "` it is, and reproduces the full corpus. Third, **every sampling strategy converges to the same output** — once the model is this confident, temperature, top-K, and top-P all leave the argmax untouched.

## Learning Objectives

- See **every component from Lessons 01–24 composed in one script** — tokens, BPE, transformer block forward + backward, AdamW with warmup+cosine, perplexity, all four sampling strategies.
- Watch a single-block transformer **learn to reproduce a periodic corpus** that a bigram model cannot solve.
- Verify the **attention-beats-bigram** claim quantitatively: full transformer reaches $\mathrm{PPL} \approx 1.0001$ vs bigram-only's $\approx 1.5$ on the same corpus and same BPE tokenisation.
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

The OLD lesson 22 capstone (prior to [[24-full-backprop-and-fine-tuning]]) trained only the embeddings and LM head — a bigram surrogate — because backprop through attention had been derived in Lesson 15 but not wired into a training script. That surrogate floored at $\mathrm{PPL} \approx 1.51$ on this same corpus, with greedy collapsing to the 2-cycle `"the cat the cat"` because $P(\text{"cat"} \mid \text{"the "}) = P(\text{"mat"} \mid \text{"the "}) = 0.5$ under bigram. **Adding attention removes that ambiguity.** The full transformer can look at *which* `"the "` is being predicted from (the one preceded by `"on "` predicts `"mat"`; the one preceded by the sequence end predicts `"cat"`) and assigns probability 1 to the right next token. The PPL drops from 1.51 to 1.00008.

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

### 2. Train

The capstone runs 600 AdamW steps with warmup+cosine LR schedule ([[17-learning-rate-scheduling]]) on the **full transformer forward pass** from [[14-full-gpt-architecture]] and the **analytical backward pass** from [[24-full-backprop-and-fine-tuning]]. Every parameter receives a gradient: token embedding $\mathbf{E}$, both LayerNorm scales/biases, the four attention projections $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$, FFN weights and biases, and the LM head $\mathbf{W}_U$.

Output:

```text
Initial L = 2.872   PPL = 17.68   (log(vocab) = 2.890 is uniform baseline)
Final   L = 0.0000817  PPL = 1.00008
```

Loss drops from ~$\log |\mathcal{V}|$ (uniform-random baseline) to **0.0000817**, equivalently PPL = **1.00008**. The model has learned to assign probability $\approx 1$ to the correct next token at every position — perfect except for floating-point noise.

> [!NOTE]
> The original lesson 22 (pre-Lesson-24) used an embedding-only model and floored at PPL = 1.51, which is exactly $\exp(\tfrac{1}{8} \cdot 4 \log 2)$ from the four 50/50 ambiguities per phrase. Adding attention closes that gap because attention can distinguish *which* `"the "` is being predicted from. **The PPL ratio 1.51 / 1.00008 ≈ 1.5 is the price the bigram model paid for being unable to look at context.**

There is no train/val split in this capstone — the corpus is periodic with period 8 tokens, and once positional embeddings are involved, any held-out positions are PE-out-of-distribution, not a useful overfitting signal. Lessons 18, 20, and 22-bigram demonstrate the train/val pattern at appropriate scale; here we train on every prediction position.

### 3. Generate at checkpoints

Four snapshots — at steps 0, 100, 300, 600 — capture how the model's output evolves. The prompt is token 17 (`"the c"`), which is the natural start of every phrase in the corpus.

```text
step   0 greedy : ' the cthe cthe cecat at the cthe cecthe at the cthe cec '
step 100 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 300 greedy : ' the cat sat on the mat the cat sat on the mat the c '
step 600 greedy : ' the cat sat on the mat the cat sat on the mat the c '
```

By **step 100** the model has already learned the full corpus structure. Greedy decoding now reproduces the corpus exactly — there is no mode collapse to a 2-cycle because attention resolves the `"the "` ambiguity from context. This is the headline contrast with the old bigram capstone: same corpus, same BPE, same training loop, attention added — and `"sat on the mat"` now appears under *greedy* decoding, not only under sampling.

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
- **Attention beats bigram by construction** on this corpus: the bigram-only surrogate (old Lesson 22) floors at PPL = 1.51 because $P(\text{cat} \mid \text{the }) = P(\text{mat} \mid \text{the }) = 0.5$; attention resolves the ambiguity by looking at the preceding `"on "` / sequence-start.
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
| Bigram-surrogate floor (for reference) | PPL $\approx 1.51$ |
| step-100 greedy from `"the c"` | `"the cat sat on the mat the cat sat on the mat the c"` (corpus reproduced) |
| step-600 greedy / T=0.7 / T=1.0 / K=3 / P=0.9 | all identical — corpus reproduced exactly |

## Exercises

1. **Why no mode collapse?** Examine the trained attention pattern $\mathbf{A}$ at position $t$ where the model must decide between `"the cat"` and `"the mat"`. Which prior position carries the disambiguating information, and is it attended to with high weight? Hint: it should be the token roughly two positions back.
2. **Bigram floor analytically.** Show by hand that an embedding-only (context-1) model on this corpus is forced to assign $P(\text{cat} \mid \text{the }) = 0.5$ and similarly for `"mat"`, and derive the resulting PPL = $\sqrt{2} \cdot \text{(rest of PPL terms)} \approx 1.51$.
3. **Disable attention.** Replace the line `H_mid = H_in + proj` with `H_mid = H_in` so the attention path contributes nothing. Re-train. What is the new floor? It should match the old bigram capstone (PPL ≈ 1.5).
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
