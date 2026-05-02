# Lesson 20: Perplexity and Evaluation

[Lesson 03](03-cross-entropy-loss.md) introduced cross-entropy as the model's training objective and information-theoretic floor; [Lesson 18](18-training-loop.md) used it as the per-step diagnostic. **Perplexity** is the same number expressed in a more interpretable unit — an "effective branching factor" — and is the universal currency for comparing language models across architectures and corpora. This lesson derives perplexity from cross-entropy, explains the relationship in both directions, and tracks perplexity through a real training run.

## Learning Objectives

- Derive **perplexity** $\mathrm{PPL} = e^{\mathcal{L}}$ from average cross-entropy $\mathcal{L}$ in nats (and $\mathrm{PPL} = 2^{\mathcal{L}}$ for $\mathcal{L}$ in bits).
- Interpret perplexity as the **effective number of equally-likely choices** the model is uncertain about per token.
- State the two **floor and ceiling** baselines: uniform-prior PPL = $|\mathcal{V}|$ (model knows nothing) and PPL = 1 (model is perfect).
- Track perplexity through a training run and recognise the same train/val signatures from [Lesson 18](18-training-loop.md): healthy convergence, underfitting, overfitting.
- Use perplexity to **compare language models** trained on the same corpus (lower = better) and discuss why cross-corpus comparison is harder.

## Background

Cross-entropy and the source-coding bound from [Lesson 03](03-cross-entropy-loss.md). The bigram model and its PPL = 1.414 reference value from [Lesson 05](05-bigram-language-model.md). Training-loop diagnostics from [Lesson 18](18-training-loop.md).

## Defining Perplexity

### Theory

Given a sequence $x_1, x_2, \dots, x_T$, a language model's average per-token cross-entropy in nats is

$$\mathcal{L} \;=\; -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_1, \dots, x_t).$$

Perplexity is its exponential:

$$\boxed{\mathrm{PPL} \;=\; \exp(\mathcal{L}) \;=\; \prod_{t=1}^{T-1} P_\theta(x_{t+1} \mid x_{<t})^{-1/(T-1)}.}$$

Two cases give clean intuition:

- **Uniform predictions** $P(x_{t+1} \mid \cdot) = 1/|\mathcal{V}|$. Then $\mathcal{L} = \log |\mathcal{V}|$ and $\mathrm{PPL} = |\mathcal{V}|$. The model has zero information beyond knowing the alphabet — it is **as uncertain as a uniform pick from the whole vocabulary**.
- **Perfect predictions** $P(x_{t+1} \mid \cdot) = 1$ for the actual next token. Then $\mathcal{L} = 0$ and $\mathrm{PPL} = 1$. The model knows the answer with certainty.

In between, $\mathrm{PPL}$ is the **effective branching factor** — if the model's uncertainty were converted to a uniform distribution, this is how many options it would pick from on average.

### Example — PPL of three reference distributions

```rustlab
% Uniform over 4 classes — PPL should equal 4
p_uniform = [0.25, 0.25, 0.25, 0.25];

% Peaked: 0.7 / 0.1 / 0.1 / 0.1 — much lower entropy
p_peaked  = [0.7,  0.1,  0.1,  0.1];

% Sharp: 0.97 / 0.01 / 0.01 / 0.01 — very confident
p_sharp   = [0.97, 0.01, 0.01, 0.01];

% L = -sum(p * log(p)) is the entropy; PPL = exp(L).
function ppl = entropy_to_ppl(p)
  L = -sum(p .* log(p));
  ppl = exp(L);
end

print("PPL uniform =", entropy_to_ppl(p_uniform), "  (should equal 4)");
print("PPL peaked  =", entropy_to_ppl(p_peaked));
print("PPL sharp   =", entropy_to_ppl(p_sharp));
```

A uniform distribution over 4 classes has entropy $\log 4 \approx 1.386$ nats and $\mathrm{PPL} = 4$. Peaked distributions have $\mathrm{PPL}$ between 1 and 4, with $\mathrm{PPL} \to 1$ as the distribution concentrates on a single class.

## Why Perplexity, Not Cross-Entropy?

### Theory

Cross-entropy and perplexity carry the same information, but perplexity has two advantages for human reading:

1. **Linear-in-uncertainty units.** A jump from PPL 50 to 25 is "the model halved its uncertainty"; the same jump in cross-entropy is from $\log 50 = 3.91$ to $\log 25 = 3.22$ — accurate but unintuitive.
2. **Direct comparison to baselines.** "Is $\mathrm{PPL} = 30$ good?" depends on $|\mathcal{V}|$. If $|\mathcal{V}| = 50000$ (GPT-3 scale), then 30 is excellent — the model is one in a thousand of the uniform baseline. If $|\mathcal{V}| = 256$ (byte-level LM), 30 is mediocre.

The conversion is one line: $\mathrm{PPL} = e^{\mathcal{L}}$ for $\mathcal{L}$ in nats, $\mathrm{PPL} = 2^{\mathcal{L}}$ for $\mathcal{L}$ in bits. Always check which logarithm a paper is using.

## Training Curves in Perplexity Units

### Theory

Take any training run and plot $\mathrm{PPL}$ instead of $\mathcal{L}$ on the y-axis. The shape of the curve is the same — both are monotone-decreasing functions of training progress — but PPL has two visually-useful properties:

- The curve **starts at $|\mathcal{V}|$** and **must end at 1 or higher**. The y-axis has natural endpoints, so a glance tells you how close you are to perfect.
- The curve is **multiplicative-rate-friendly**. If $\mathrm{PPL}_t / \mathrm{PPL}_{t+1} = 1.05$ for every $t$, the model is improving by a constant factor per step — easy to recognise on a log-scale plot.

The standalone script `perplexity_curve.r` reruns the [Lesson 18](18-training-loop.md) training loop with PPL computed on both train and val sets and plots the result.

### Example — PPL endpoint sanity for the bigram corpus

```rustlab
% From Lesson 05: bigram H = 0.347 nats on the period-4 abcb corpus.
H = 0.347;
PPL_bigram = exp(H);
print("Bigram PPL on 'abcbabcb...' (Lesson 05): ", PPL_bigram);
print("Vocab size 3 -> uniform PPL = 3.");
```

A trained bigram on the period-4 corpus reaches $\mathrm{PPL} \approx 1.414$ — the theoretical floor. A model that did *better* would have to be using more than the previous-token context (e.g. the trigram `(prev2, prev)` would push PPL all the way to 1, since the corpus is fully deterministic given two tokens).

## Comparing Models

### Theory

When comparing two language models on the same test set the rule is simple:

$$\text{Model A is better than Model B}\iff \mathrm{PPL}_A < \mathrm{PPL}_B.$$

Practical caveats:

- **Same vocabulary.** A model with a larger vocab can artificially inflate or deflate PPL. Reasonable comparisons normalise to *bits per byte* or *bits per character* — invariants under tokenisation choice.
- **Same test set.** PPL on Wikipedia is incomparable to PPL on Python source code; they are different distributions. Reported "GPT-3 PPL on PTB" is a contract: the test corpus is fixed.
- **Length normalisation.** PPL implicitly averages over sequence length. A model that does well on short sequences but poorly on long ones can have similar PPL to a model with the opposite profile — break the average down by length when comparing for end-task suitability.

### Example — PPL after one round of training

The standalone script trains the same 24-parameter embedding+head model from [Lesson 18](18-training-loop.md) for 600 steps and computes the train/val PPL at each logged step. Expected behaviour:

- **Step 0:** $\mathrm{PPL} \approx 3$ (uniform-prior baseline for $|\mathcal{V}| = 3$).
- **Step 600:** $\mathrm{PPL} \approx 1.4$ (the bigram floor on the period-4 corpus).

The training run pushes both train and val PPL down by roughly $3 \to 1.4$ — almost halving the effective branching factor.

## Connection to Compression

### Theory

Cross-entropy in bits per token is **exactly** the compressed size (in bits per source symbol) achieved by an arithmetic coder using the model's predicted distribution. This means:

- A language model with $\mathrm{PPL} = 2^{2.3} = 4.92$ on a corpus reaches compression ratio $2.3$ bits per character.
- Halving the perplexity gains 1 bit per character.
- The theoretical floor for any next-token model is the corpus's true conditional entropy $H(X_t \mid X_{<t})$, which corresponds to $\mathrm{PPL} = e^{H}$.

"Better language model" and "better text compressor" are not analogies — they are exactly the same number under arithmetic coding ([Lesson 05](05-bigram-language-model.md)). A 1-bit-per-char improvement in perplexity translates directly to half-the-storage on a Wikipedia dump.

## Per-Token Perplexity Distribution

### Theory

Mean PPL is a single summary number, but the per-token distribution carries useful diagnostic information:

- **Bimodal PPL** — many tokens at PPL ≈ 1 (the model knows them) and a few at very high PPL. Indicates the model has learned the easy patterns but fails on a specific class of inputs.
- **Heavy-tailed PPL** — a few tokens with PPL in the thousands dominate the mean. These are usually rare-vocabulary or ambiguous-context tokens.
- **Tight unimodal PPL** — the model is uniformly imperfect, no specific failure pattern.

A useful evaluation isn't just "what is the average PPL?" but "what is the distribution of $-\log P_\theta(x_{t+1})$ across the test set?" Plotting that histogram is the cheapest diagnostic that a model is working as intended.

## Connection to Earlier Lessons

### Theory

- **Lesson 03** introduced cross-entropy as $\mathcal{L} = -\sum p \log q$ for distributions, and as the average $-\log P(x_{t+1} \mid x_{<t})$ for a sequence. PPL is just $e^{\mathcal{L}}$ on the same average.
- **Lesson 05** computed the bigram model's reference PPL at $\sim 1.414$ for the period-4 corpus and noted Markov-1 cannot beat it.
- **Lesson 14** counted parameters; the Chinchilla scaling laws relate parameters → tokens → $\mathrm{PPL}$ via empirically-fitted curves.
- **Lesson 18** plotted $\mathcal{L}$ over training; this lesson plots $e^{\mathcal{L}}$ on top of the same data.

## Key Takeaways

- $\mathrm{PPL} = e^{\mathcal{L}}$ for $\mathcal{L}$ in nats; $\mathrm{PPL} = 2^{\mathcal{L}}$ for $\mathcal{L}$ in bits. Always check the log base.
- Perplexity is the **effective branching factor**: how many equally-likely options the model is uncertain about per token.
- Reference points: $\mathrm{PPL} = |\mathcal{V}|$ (uniform random) and $\mathrm{PPL} = 1$ (perfect).
- Lower PPL = better model — but only on the same vocab and the same test set.
- $\mathrm{PPL}$ in bits is **literally** a compression ratio; LM quality and text-compression ratio are the same number.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `perplexity_basics.r` | PPL of three reference distributions (uniform, peaked, sharp) and prints their endpoints |
| `perplexity_curve.r` | re-runs the Lesson 18 24-parameter LM training loop and plots train/val PPL per step (alongside the loss curve) |

Run all with `make lesson-20` (or `rustlab run lessons/20-perplexity-and-evaluation/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `entropy_to_ppl(p_uniform)` over 4 classes | `4.0` |
| `entropy_to_ppl(p_peaked)` (0.7/0.1/0.1/0.1) | ≈ `2.32` |
| `entropy_to_ppl(p_sharp)` (0.97/0.01/0.01/0.01) | ≈ `1.21` |
| Initial train PPL (`perplexity_curve.r`) | ≈ `3.0` (uniform over 3 classes) |
| Final train PPL (`perplexity_curve.r`) | ≈ `1.4` (the bigram floor) |
| Final val PPL | similar to train PPL (no overfit on this small model) |

## Exercises

1. **Endpoint sanity.** What perplexity does a random-init model produce on a $|\mathcal{V}| = 256$ byte-level vocabulary? On $|\mathcal{V}| = 50000$ BPE tokens? Are these the right baselines to compare a trained LM against?
2. **Bits per character.** GPT-3 reaches roughly 1.7 bits per byte on a typical English corpus. Convert this to perplexity over the byte vocabulary ($|\mathcal{V}| = 256$). What fraction of the uniform PPL is that?
3. **Cross-vocab comparison.** Two models — one with $|\mathcal{V}| = 100$ and another with $|\mathcal{V}| = 1000$ — claim PPL = 30 on the same English text. Are they equally good? Compute bits-per-character for each and explain.
4. **Tail-of-the-distribution.** Take any trained model on a fixed test set. Its mean per-token loss is $\mathcal{L}$, but the *median* token loss is often much lower. Why? What does this say about which tokens dominate the average?
5. **Floor for the period-4 corpus.** A trigram model conditions on the previous *two* tokens. Compute its theoretical PPL on the corpus `"abcbabcb…"`. Is it equal to 1, greater than 1, or undefined?

## What's next

Phase 7 closes here. With BPE in hand for tokenisation and PPL as the evaluation metric, every measurement an LLM publication uses is now derivable from first principles. Phase 8 (Lessons 21–22) covers **generation strategies** — greedy, temperature, top-K, top-P sampling — and the **capstone**: a single end-to-end script that trains a small GPT and generates text at every checkpoint. After that, you have seen every component of GPT built up from scratch.
