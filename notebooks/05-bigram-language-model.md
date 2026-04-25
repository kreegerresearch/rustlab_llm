# Lesson 05: The Bigram Language Model

A language model assigns a probability to every possible next token. The **bigram model** makes the simplest possible assumption: the next token depends only on the current one. Despite its simplicity, it introduces the full train-then-sample loop that every neural language model follows.

## Learning Objectives

- Define a **bigram language model** and explain how it estimates next-token probabilities from counts.
- Construct and interpret a **bigram frequency matrix** from a corpus.
- Normalise count rows into valid probability distributions.
- Trace through the **sampling algorithm** (cumulative distribution + uniform threshold) step by step.
- Explain why a bigram model is a useful baseline before introducing neural approaches.

## Background

Tokens and vocabulary from [Lesson 01](01-tokens-and-encoding.md). Probability distributions and softmax from [Lesson 02](02-probability-and-softmax.md). Cross-entropy loss from [Lesson 03](03-cross-entropy-loss.md) (used to define the training objective; not strictly required to follow the count-based estimator).

## The Language Modelling Problem

The goal is to estimate

$$P(x_{t+1} \mid x_1, x_2, \ldots, x_t).$$

The full history can be arbitrarily long. The bigram model makes the **Markov assumption**:

$$P(x_{t+1} \mid x_1, \ldots, x_t) \approx P(x_{t+1} \mid x_t).$$

This reduces the problem to estimating $|\mathcal{V}|^2$ conditional probabilities — one for every ordered pair of tokens.

## Building the Bigram Matrix

**Step 1 — Count.** Scan the corpus and tally every consecutive pair $(x_t, x_{t+1})$ into the count matrix $\mathbf{C} \in \mathbb{Z}_{\geq 0}^{|\mathcal{V}| \times |\mathcal{V}|}$:

$$C_{ij} = \text{number of times token } j \text{ follows token } i.$$

**Step 2 — Normalise.** Convert each row to a probability:

$$P_{ij} = \frac{C_{ij}}{\sum_{k=1}^{|\mathcal{V}|} C_{ik}}.$$

Build this for the corpus `"abcbabcba"`:

<!-- hide -->
```rustlab
% Corpus: "abcbabcba" — vocabulary a=1, b=2, c=3
vocab_size = 3;
seq = [1, 2, 3, 2, 1, 2, 3, 2, 1];
tokens = {"a", "b", "c"};
```

```rustlab
n_tokens  = len(seq);
n_bigrams = n_tokens - 1;

% Build the count matrix
C = zeros(vocab_size, vocab_size);
for t = 1:(n_tokens - 1)
  i = seq(t);
  j = seq(t + 1);
  C(i, j) = C(i, j) + 1;
end

print("Bigram count matrix C  (row=current, col=next):");
print(C);
```

The corpus has ${n_tokens} tokens, producing ${n_bigrams} bigrams. Total counts sum to ${sum(reshape(C, 1, vocab_size * vocab_size))} — matches $n_{\text{bigrams}}$.

```rustlab
% Row-normalise to probability matrix P
P = zeros(vocab_size, vocab_size);
for i = 1:vocab_size
  row_sum = sum(C(i));
  P(i, 1) = C(i, 1) / row_sum;
  P(i, 2) = C(i, 2) / row_sum;
  P(i, 3) = C(i, 3) / row_sum;
end

print("Normalised probability matrix P:");
print(P);

row_sums = [sum(P(1)), sum(P(2)), sum(P(3))];
print("Row sums (each should be 1):", row_sums);
```

Token `a` always goes to `b`. Token `c` always goes to `b`. Token `b` goes to either `a` or `c` with equal probability.

### Laplace Smoothing

If a pair never appeared, $P_{ij} = 0$ and the cross-entropy loss is infinite on that bigram. **Add-one smoothing** ensures every pair has non-zero probability:

$$P_{ij}^{\text{smooth}} = \frac{C_{ij} + 1}{\sum_k (C_{ik} + 1)} = \frac{C_{ij} + 1}{\sum_k C_{ik} + |\mathcal{V}|}.$$

```rustlab
C_smooth = C + ones(vocab_size, vocab_size);
P_smooth = zeros(vocab_size, vocab_size);
for i = 1:vocab_size
  row_sum_s = sum(C_smooth(i));
  P_smooth(i, 1) = C_smooth(i, 1) / row_sum_s;
  P_smooth(i, 2) = C_smooth(i, 2) / row_sum_s;
  P_smooth(i, 3) = C_smooth(i, 3) / row_sum_s;
end

print("Laplace-smoothed probability matrix P_smooth:");
print(P_smooth);

min_smooth = min(reshape(P_smooth, 1, vocab_size * vocab_size));
```

Every entry in $P^{\text{smooth}}$ is now $\geq ${min_smooth:%.3f}$ — no more zero-probability bigrams.

### Row Entropy

The entropy of each row tells us how predictable the next token is:

```rustlab
eps = 1e-12;
H = zeros(vocab_size);
for i = 1:vocab_size
  p = P(i);
  H(i) = max([0.0, -sum(p .* log2(p + eps))]);
end
```

Row entropies: $H(a) = ${H(1):%.3f}$ bits (deterministic → `b`), $H(b) = ${H(2):%.3f}$ bits (max for 2 equal options), $H(c) = ${H(3):%.3f}$ bits (deterministic → `b`).

```rustlab
figure()
imagesc(C, "viridis")
title("Bigram Count Matrix C (a,b,c)")
```

```rustlab
figure()
imagesc(P, "viridis")
title("Bigram Probability Matrix P (row-normalised)")
```

## Sampling from the Model

To generate text, repeatedly sample the next token using the **CDF method**:

1. Look up the row $\mathbf{P}_i$ for the current token.
2. Compute the cumulative distribution: $\text{CDF}_j = \sum_{k=1}^{j} P_{ik}$.
3. Draw $u \sim \text{Uniform}(0, 1)$.
4. The sampled token is $x_{t+1} = \min\{j : \text{CDF}_j \geq u\}$.

```rustlab
% Use the probability matrix from above
P = [0.0, 1.0, 0.0; 0.5, 0.0, 0.5; 0.0, 1.0, 0.0];

print("Sampling mechanism demonstration:");
p_b = P(2);
cdf_b = cumsum(p_b);
print("  P(b)   =", p_b);
print("  CDF(b) =", cdf_b);
print("  u=0.3 -> sum(CDF < 0.3) + 1 =", sum(cdf_b < 0.3) + 1, "  (expect 1 = a)");
print("  u=0.7 -> sum(CDF < 0.7) + 1 =", sum(cdf_b < 0.7) + 1, "  (expect 3 = c)");
```

### Generate a sequence

```rustlab
n_generate = 12;
generated = zeros(n_generate);
generated(1) = 1;   % start with token a

% Hand-set draws so the rendered output reproduces bit-for-bit.
% TODO: replace with `draws = rand(n_generate - 1)` once rustlab adds an RNG seed API.
draws = [0.42, 0.71, 0.13, 0.66, 0.28, 0.85, 0.19, 0.93, 0.51, 0.37, 0.64];

for t = 1:(n_generate - 1)
  curr = generated(t);
  cdf = cumsum(P(curr));
  generated(t + 1) = sum(cdf < draws(t)) + 1;
end

print("Uniform draws:", draws);
print("Generated sequence (indices):", generated);
```

Notice the structure: `b` appears at every even position, because both `a` and `c` transition deterministically to `b`.

### Training Loss and Perplexity

The count-based bigram model is the **maximum likelihood estimator** of the Markov model:

$$\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_{x_t, x_{t+1}}.$$

The **perplexity** of the model on a sequence is $\exp(\mathcal{L})$ — the model's "effective branching factor" at each step.

```rustlab
log_probs = [log(1.0), log(0.5), log(1.0), log(0.5), log(1.0), log(0.5), log(1.0), log(0.5)];
mean_ce = -real(mean(log_probs));
ppl = exp(mean_ce);
```

Mean cross-entropy on the training corpus: ${mean_ce:%.4f}$ nats, corresponding to perplexity ${ppl:%.3f}$.

```rustlab
figure()
subplot(3, 1, 1)
bar(tokens, P(1), "P(next | a) — deterministic: always b")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 2)
bar(tokens, P(2), "P(next | b) — equal: a or c")
ylabel("Probability")
ylim([0, 1])

subplot(3, 1, 3)
bar(tokens, P(3), "P(next | c) — deterministic: always b")
xlabel("Next token")
ylabel("Probability")
ylim([0, 1])
```

## Key Takeaways

- The bigram model encodes corpus statistics in a single matrix — reading a row tells you exactly what the model "thinks" will follow a given token.
- The CDF-based sampling algorithm is foundational — it reappears in every sampling strategy in Lesson 21 (top-K, nucleus).
- A bigram model is fast, interpretable, and sets a performance floor. Any neural model should beat it to justify its cost.
- Its weakness: context is limited to one token. It cannot model dependencies spanning more than two positions.

## Standalone Scripts

| Script | What it computes |
|---|---|
| `bigram_counts.r` | the 3×3 bigram count + normalised + smoothed probability matrices for `"abcbabcba"`; heatmaps |
| `bigram_sampling.r` | CDF-based sampling trace plus a 12-token generated sequence |

Run all with `make lesson-05` (or `rustlab run lessons/05-bigram-language-model/<name>.r`).

## Expected Numerical Outputs Summary

| Variable | Expected Value |
|---|---|
| `n_tokens` | `9` |
| `n_bigrams` | `8` |
| `sum(C)` | `8` (= `n_bigrams`) |
| `P(1, 2)` ($P(b\mid a)$) | `1.0` |
| `P(2, 1)` ($P(a\mid b)$) | `0.5` |
| `P(2, 3)` ($P(c\mid b)$) | `0.5` |
| `min_smooth` | ≈ `0.143` (= 1/7) |
| `H(1)`, `H(3)` | `0` bits (deterministic) |
| `H(2)` | `1` bit (max for 2 equal options) |
| `mean_ce` | ≈ `0.347` nats |
| `ppl` | ≈ `1.414` |

## Exercises

1. **Entropy of rows.** The notebook prints $H(a), H(b), H(c)$ for `"abcbabcba"`. Which token is most predictable? Which is least? Repeat for the corpus `"aabbcc"`.
2. **Laplace smoothing effect.** For `"abcbabcba"` with $|\mathcal{V}| = 3$, compute the Laplace-smoothed probability for the bigram (a → a), which never appeared. What is $P^{\text{smooth}}_{11}$?
3. **Longer corpus.** Extend the corpus to `"abcbabcbabc"` and recompute the bigram matrix. Do the probabilities change significantly? What happens as the corpus grows?
4. **Perplexity interpretation.** What does a perplexity of 1 mean? What about perplexity of $|\mathcal{V}|$?
5. **Beyond bigrams.** A trigram model conditions on the last two tokens: $P(x_{t+1} \mid x_{t-1}, x_t)$. How many parameters does a trigram model have for vocabulary size $|\mathcal{V}|$? How does this scale compared to the bigram model?

## What's next

Lesson 06 replaces the count table with a **learned linear layer** $y = Wx + b$ and shows how **gradient descent** drives the parameters toward the same optimum the count-based estimator finds analytically. The transition from "look up" to "learn" is the conceptual jump from a classical n-gram model to a neural one — every subsequent lesson uses gradient descent.
