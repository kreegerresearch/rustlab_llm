# Lesson 22: Putting It All Together

This is the capstone. Every prior lesson built one piece — characters, softmax, embeddings, attention, the transformer block, AdamW, perplexity, sampling, the KV cache. This lesson puts them in one script and watches a tiny language model **learn**.

The corpus is the periodic phrase `"the cat sat on the mat "` repeated four times. After BPE tokenisation, training, and sampling, the model produces output that is visibly text-like:

```text
step   0 greedy  :   e eat ttheat ttheat t
step 100 greedy  :   at sat sat sat sat sat s
step 300 greedy  :   at the cat the cat the cat the cat
step 800 greedy  :   at the cat the cat the cat the cat
step 800 T=0.7   :   at sat on the mat sat the cat s
step 800 T=1.0   :   e at the cat the cat the cat sat
step 800 K=3     :   at the cat the cat on the mat
step 800 P=0.9   :   mat sat the cat on the mat
```

Three things to read from that gallery: (a) loss is dropping (random → bigram-correct), (b) greedy **mode-collapses** to `"the cat"` because at the `"the"` token the model splits evenly between `"cat"` and `"mat"`, and (c) **sampling recovers the full corpus structure** — `"sat on the mat"` appears once temperature is on. This is the same dynamic as the period-4 abcb corpus from [[18-training-loop]] and [[21-sampling-and-generation]], scaled up to word tokens.

## Learning Objectives

- See **every component from Lessons 01–21 composed in one script** — tokens, BPE, embeddings, training, AdamW, schedule, perplexity, sampling.
- Read training curves (loss, PPL, gradient norm, learning rate) and explain what each is doing.
- Verify the **bigram-floor sanity check** ([[20-perplexity-and-evaluation]]) on a word-level corpus.
- Recognise the **mode-collapse → sampling-recovery** pattern at the word level.
- Identify what is **trained end-to-end here** vs. what was built in earlier lessons but left frozen for tractability.

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

The capstone script `capstone.rlab` references each by section header.

## What This Lesson Trains End-to-End

For tractability the capstone trains the **embeddings + LM head** ([[18-training-loop]]'s 24-parameter model, scaled up to a BPE vocabulary of size 18 with $d_{\text{model}} = 6$ — 216 parameters total). This is enough to learn the bigram structure of the corpus *at the word level*, which is the interesting thing after BPE merges produce word-like tokens.

What is **shown in the curriculum but not retrained here**:

- The transformer block ([[13-transformer-block]]). The full forward pass — multi-head attention + FFN + LayerNorm + residuals — was implemented and verified in lessons 08–13, and stacked in lesson 14. Adding it to the trained model requires the full backward path ([[15-backpropagation]]) through attention; that is the line where this curriculum stops and a production codebase begins.
- The KV cache ([[21-sampling-and-generation]]). It is mathematically tied to attention, so it only buys speed when attention is present. The cache derivation, equivalence proof, and FLOP analysis live in lesson 21.

> [!IMPORTANT]
> "Mini-GPT" here means **the GPT recipe applied to a small model**. The recipe is everything in the table above. The model is small because rustlab is an interpreter and the goal is comprehension, not throughput. Every choice (vocabulary size, depth, training set) scales transparently to a real LLM — only the numbers change.

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

A standard [[18-training-loop]] over 800 steps. Forward pass: $h_t = \mathbf{E}(x_t, :)$; $p_t = \mathrm{softmax}(h_t \mathbf{W})$; $\mathcal{L} = -\log p_t(x_{t+1})$. Backward: $d\mathbf{logits} = p_t - \mathbf{1}_{x_{t+1}}$; the rest is two matrix multiplies and a row scatter. AdamW ([[16-adamw-optimizer]]) with warmup + cosine schedule ([[17-learning-rate-scheduling]]).

Output:

```text
Initial train loss: 2.969  PPL: 19.48   (log(vocab) = 2.890 is uniform baseline)
Final   train loss: 0.412  PPL:  1.51
Final   val   loss: 0.314  PPL:  1.37
```

Both train and val PPL drop from ~19 (uniform over 18 tokens) to **1.37–1.51** — close to the conditional-entropy floor for this corpus. The val PPL being slightly *lower* than train PPL is a quirk of the small val set (7 pairs); on real corpora train and val track each other closely until overfitting kicks in.

### 3. Generate at checkpoints

Four snapshots — at steps 0, 100, 300, 800 — capture how the model's output evolves:

```text
step   0 greedy :   e eat ttheat ttheat t                  % random
step 100 greedy :   at sat sat sat sat sat s               % learned "sat" only
step 300 greedy :   at the cat the cat the cat the cat    % learned bigrams; mode-collapsed
step 800 greedy :   at the cat the cat the cat the cat    % converged
```

Greedy decoding picks the argmax at every step. At step 300 the model has learned that `"the "` is followed by either `" c"` (→ `"the cat"`) or `" m"` (→ `"the mat"`) — each with probability 0.5. Greedy picks one and stays with it, producing the 2-cycle `"the cat the cat …"`.

Switching to **temperature sampling** ([[21-sampling-and-generation]]) recovers the full corpus structure:

```text
step 800 T=0.7  :   at sat on the mat sat the cat s
step 800 T=1.0  :   e at the cat the cat the cat sat
step 800 K=3    :   at the cat the cat on the mat
step 800 P=0.9  :   mat sat the cat on the mat
```

`"sat on the mat"` appears unprompted — the model has captured the full corpus phrase and sampling lets both alternative branches occur. This is the same observation from [[21-sampling-and-generation]]'s gallery, in a richer setting.

### 4. Read the diagnostics

The capstone saves a 4-panel diagnostic figure (`capstone.svg`):

- **Loss curve** (train vs val). Both fall together — healthy run.
- **PPL curve** — same shape; gives the more interpretable "effective branching factor" units.
- **Gradient norm** — drops from $\sim 10^{-1}$ to $\sim 10^{-3}$ over training. The healthy "decrease over time" signature from [[18-training-loop]].
- **Learning rate** — the warmup + cosine envelope from [[17-learning-rate-scheduling]].

These are the four plots a production training run would also show. Reading them is the same — the only difference is the scale on the y-axis.

## What Production Adds

This curriculum stops at the boundary between "every component understood" and "every component trained end-to-end." A production transformer trainer adds:

- **Backprop through the transformer block.** [[15-backpropagation]] derived the chain rule through attention and FFN; production code wires it into the optimiser step.
- **Mini-batching.** The capstone uses full-batch gradient descent on 24 pairs. Real training samples random sequences from a corpus of billions of tokens, $O(10^4$–$10^6)$ tokens per batch.
- **Mixed precision** (fp16 / bf16). Forward in low precision, accumulate in fp32. Cuts memory and time by ~2×.
- **Gradient clipping.** A `max_norm = 1.0` clip step ([[18-training-loop]]'s sidebar) prevents single-step divergence on outlier batches.
- **Dropout.** Stochastic activation zeroing during training ([[13-transformer-block]]'s sidebar) — small effect on small models, important at scale.
- **Distributed training.** Data-parallel and tensor-parallel splits across many GPUs; the only architectural addition is the inter-GPU all-reduce on gradients.
- **A real corpus.** Wikipedia + Common Crawl + code, hundreds of billions of tokens, tokenised once and held in a memory-mapped file.

Every one of those is an engineering layer over the math you have already built. The reverse — a production trainer that wires up all that infrastructure but gets the math wrong — fails silently and is far harder to debug. **The understanding compounds the way the model's loss does.**

## Key Takeaways

- The capstone composes Lessons 01–21 into a single 250-line script that **trains and generates** end-to-end.
- Both train and val PPL drop from ~19 (uniform over 18 tokens) to ~1.4–1.5, near the bigram floor.
- Greedy decoding mode-collapses; temperature / top-K / top-P sampling recover the corpus structure. The pattern is identical to [[21-sampling-and-generation]]'s smaller demo.
- The four diagnostic plots (loss, PPL, gradient norm, LR) are exactly what a production run reports.
- Scaling to a real LLM swaps the model size, the corpus, the precision, and the training scale — not the underlying math.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `capstone.rlab` | Full end-to-end: char-tokenise corpus → BPE merge (Lesson 19) → train embedding+head (Lesson 18) → checkpoint generation (Lesson 21) → diagnostic plots (Lessons 17, 18, 20) |

Run with `make lesson-22` (or `rustlab run lessons/22-putting-it-all-together/capstone.rlab`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| Char vocab size | $10$ |
| Initial char sequence length | $92$ ($23 \times 4$) |
| Number of BPE merges | $8$ |
| Final vocab size | $18$ |
| Final token sequence length | $32$ |
| Trainable parameters | $216$ (vocab × $d_{\text{emb}}$ × 2) |
| Initial train loss | $\approx 2.97$ (≈ $\log(18) = 2.89$ baseline) |
| Final train loss | $\approx 0.41$ |
| Final train PPL | $\approx 1.51$ |
| Final val PPL | $\approx 1.37$ |
| step-800 greedy from `' '` | `" at the cat the cat the cat the cat "` (mode collapse) |
| step-800 T=0.7 from `' '` | text containing `"sat"`, `"the cat"`, `"the mat"` |

## Exercises

1. **Bigram floor.** Compute the true conditional entropy $H(X_{t+1} \mid X_t)$ of the BPE-tokenised corpus by hand. Does it match the trained-model train PPL? Where does the gap come from?
2. **More merges.** Re-run the capstone with $n_{\text{merges}} = 20$ instead of 8. What happens to the vocab size, sequence length, and final PPL? Why does PPL eventually rise once merges exceed the corpus's natural word boundaries?
3. **Longer corpus.** Change `n_reps` from 4 to 10. How does the train/val PPL gap evolve? At what `n_reps` does the validation set become large enough that the train/val curves visually coincide?
4. **Repetition penalty at the word level.** Add repetition penalty $\rho = 1.5$ over a window of 4 tokens to the step-800 greedy decoder. Does it break out of the `"the cat"` mode collapse to recover `"the mat"`?
5. **Capstone with a different corpus.** Replace `"the cat sat on the mat "` with `"a quick brown fox "` (also 23 chars after a trailing space). Re-run the capstone. Identify the new bigram split point (analogous to `"the → cat"` vs `"the → mat"`) and verify that sampling at $T = 0.7$ recovers both branches.

## What's next

**This is the end of the curriculum.** You have built every component of a GPT-style language model from first principles: probability and softmax, embeddings and similarity, attention and multi-head attention, positional encoding, the transformer block, the full stack, backpropagation, AdamW, learning-rate scheduling, the training loop, BPE tokenisation, perplexity, sampling strategies, and the KV cache. The capstone composes them.

The natural next steps outside the curriculum:

- **Implement backprop through the transformer block.** [[15-backpropagation]] has the chain rule; wire it into [[18-training-loop]]'s loop. The result is a model that *trains* the attention block too, not just the embeddings.
- **Scale up.** Move to a real corpus (TinyShakespeare, Wikipedia, code); increase $d_{\text{model}}$, $N$, and $|\mathcal{V}|$. The infrastructure layer is everything in [What Production Adds](#what-production-adds).
- **Read [nanoGPT](https://github.com/karpathy/nanoGPT).** The 300 lines of Python in `nanoGPT/model.py` are exactly the architecture in [[14-full-gpt-architecture]], written in PyTorch. You now know what every line means.

The math you built here is the math every modern language model runs on — the rest is engineering.
